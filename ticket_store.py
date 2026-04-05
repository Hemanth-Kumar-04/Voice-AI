"""
ticket_store.py
──────────────
Simple in-memory ticket store.  NO database – just a Python list with a lock
so it is safe when the bot runs in an async context.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ─── In-memory storage ────────────────────────────────────────────────────────
_DB_PATH = Path(__file__).resolve().parent / "db.json"
_tickets: list[dict] = []
_lock = asyncio.Lock()


def _load_tickets() -> None:
    if not _DB_PATH.exists():
        return
    try:
        data = json.loads(_DB_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            _tickets.extend(data)
    except (OSError, json.JSONDecodeError):
        # Leave tickets empty if db.json is unreadable.
        pass


def _save_tickets() -> None:
    _DB_PATH.write_text(json.dumps(_tickets, indent=2), encoding="utf-8")


_load_tickets()


# ─── Public API ───────────────────────────────────────────────────────────────
async def create_ticket(
    name: str,
    flat: str,
    issue: str,
    location: str,
    severity: str,
    started_when: str,
) -> dict:
    """Create and store a new maintenance ticket.  Returns the saved ticket."""
    ticket = {
        "id": f"issue_{uuid.uuid4().hex[:6]}",
        "name": name,
        "flat": flat,
        "issue": issue,
        "location": location,
        "severity": severity.lower(),
        "started_when": started_when,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }
    async with _lock:
        _tickets.append(ticket)
        _save_tickets()
    return ticket


async def update_ticket(ticket_id: str, updates: dict) -> Optional[dict]:
    """Update an existing ticket by ID.  Returns updated ticket or None."""
    async with _lock:
        for ticket in _tickets:
            if ticket["id"] == ticket_id:
                ticket.update(updates)
                ticket["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_tickets()
                return ticket
    return None


def get_all_tickets() -> list[dict]:
    """Return a snapshot of all tickets (sync, safe to call from endpoints)."""
    return list(_tickets)


def get_ticket_by_id(ticket_id: str) -> Optional[dict]:
    for t in _tickets:
        if t["id"] == ticket_id:
            return t
    return None
