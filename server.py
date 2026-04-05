"""
server.py
─────────
FastAPI server that:
  1. Generates LiveKit tokens for the frontend user (/token)
  2. Generates a bot token and launches bot.py as a subprocess (/start-bot)
  3. Exposes the in-memory tickets for inspection (/tickets)

Run with:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import timedelta

import jwt  # PyJWT — already a transitive dep, no new install needed
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from ticket_store import get_all_tickets, get_ticket_by_id

logger = logging.getLogger("server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AquaBot – Plumbing Voice Agent",
    version="1.0.0",
    description="Low-latency voice AI for gated-community plumbing helpdesk",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prototype only — lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_token(identity: str, room_name: str, ttl_hours: int = 2) -> str:
    """
    Generate a LiveKit participant JWT without using livekit-api.
    LiveKit tokens are standard HS256 JWTs — we build them directly
    to avoid the namespace collision between pipecat's livekit client
    and the livekit-api server SDK.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set in .env")

    now = int(time.time())
    payload = {
        "iss": api_key,           # issuer = API key
        "sub": identity,          # subject = participant identity
        "iat": now,               # issued at
        "nbf": now,               # not before
        "exp": now + ttl_hours * 3600,
        "name": identity,
        "video": {                # LiveKit grants
            "roomJoin": True,
            "room": room_name,
            "canPublish": True,
            "canSubscribe": True,
        },
    }
    token = jwt.encode(payload, api_secret, algorithm="HS256")
    # PyJWT >= 2.0 returns str, older versions return bytes
    return token if isinstance(token, str) else token.decode("utf-8")


# Track active bot subprocess per room so we don't double-spawn
_active_bots: dict[str, subprocess.Popen] = {}
_active_bot_logs: dict[str, object] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    identity: str = "resident"
    room_name: str | None = None


class StartBotRequest(BaseModel):
    room_name: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AquaBot"}


@app.post("/token")
async def get_token(req: TokenRequest):
    """
    Issue a LiveKit token for the frontend user.
    The frontend uses this to connect to the room via LiveKit JS SDK.
    """
    room_name = req.room_name or os.getenv("ROOM_NAME", "plumbing-support")
    try:
        token = _make_token(req.identity, room_name)
    except Exception as exc:
        logger.exception("Token generation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "token": token,
        "room_name": room_name,
        "livekit_url": os.getenv("LIVEKIT_URL"),
    }


@app.post("/start-bot")
async def start_bot(req: StartBotRequest):
    """
    Spawn bot.py as a subprocess for the given room.
    Generates a bot-identity token and passes it to the subprocess.
    """
    room_name = req.room_name or os.getenv("ROOM_NAME", "plumbing-support")

    # Don't spawn a second bot for the same room
    existing = _active_bots.get(room_name)
    if existing and existing.poll() is None:
        return {"status": "already_running", "room_name": room_name}

    room_url = os.getenv("LIVEKIT_URL")
    if not room_url:
        raise HTTPException(status_code=500, detail="LIVEKIT_URL not set in .env")

    try:
        bot_token = _make_token("aquabot", room_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"bot_{room_name}.log")
        log_file = open(log_path, "a", encoding="utf-8")

        proc = subprocess.Popen(
            [
                sys.executable,
                "bot.py",
                "--room-url", room_url,
                "--token", bot_token,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _active_bots[room_name] = proc
        _active_bot_logs[room_name] = log_file
        logger.info(
            "Bot spawned for room '%s' (pid=%d) log=%s",
            room_name,
            proc.pid,
            log_path,
        )
    except Exception as exc:
        logger.error("Failed to spawn bot: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to spawn bot: {exc}")

    return {
        "status": "started",
        "room_name": room_name,
        "pid": proc.pid,
        "log_path": log_path,
    }


@app.get("/tickets")
async def list_tickets():
    """Return all in-memory maintenance tickets (for debugging)."""
    return JSONResponse(content=get_all_tickets())


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.delete("/bot/{room_name}")
async def stop_bot(room_name: str):
    """Terminate the bot subprocess for a room."""
    proc = _active_bots.get(room_name)
    if not proc or proc.poll() is not None:
        raise HTTPException(status_code=404, detail="No active bot for this room")
    proc.terminate()
    log_file = _active_bot_logs.pop(room_name, None)
    if log_file:
        log_file.close()
    return {"status": "terminated", "room_name": room_name}
