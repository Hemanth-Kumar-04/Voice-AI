"""
bot.py
──────
Main Pipecat voice bot.

Pipeline flow:
  LiveKit (mic audio)
    → GroqSTTService          (Whisper speech-to-text)
    → context_aggregator.user()
    → GroqLLMService          (Llama 3.3 — OpenAI-compatible tools)
    → CartesiaTTSService      (streaming TTS)
    → LiveKit (speaker audio)
    → context_aggregator.assistant()

VAD, turn-taking, and barge-in interruption are handled entirely by Pipecat.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("bot")

# ── Pipecat core ──────────────────────────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMMessagesFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)

# ── Pipecat services ──────────────────────────────────────────────────────────
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService

# ── LiveKit transport ─────────────────────────────────────────────────────────
from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport

# ── Project modules ───────────────────────────────────────────────────────────
from prompts import SYSTEM_PROMPT
from tools import TOOLS, register_tools


# ─────────────────────────────────────────────────────────────────────────────
# Bot entry point
# ─────────────────────────────────────────────────────────────────────────────

class _TimingState:
    def __init__(self) -> None:
        self.last_user_final_ms: Optional[float] = None
        self.last_user_text: Optional[str] = None
        self.last_tts_start_ms: Optional[float] = None
        self.turn_index: int = 0


class _UserTimingProbe(FrameProcessor):
    def __init__(self, state: _TimingState) -> None:
        super().__init__(name="UserTimingProbe")
        self._state = state

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.finalized:
            now_ms = time.perf_counter() * 1000.0
            self._state.last_user_final_ms = now_ms
            self._state.last_user_text = frame.text
            logger.info("Timing: user_final turn=%s text=%r", self._state.turn_index + 1, frame.text)

        await self.push_frame(frame, direction)


class _BotTimingProbe(FrameProcessor):
    def __init__(self, state: _TimingState) -> None:
        super().__init__(name="BotTimingProbe")
        self._state = state

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            now_ms = time.perf_counter() * 1000.0
            self._state.last_tts_start_ms = now_ms
            self._state.turn_index += 1
            if self._state.last_user_final_ms is not None:
                delta_ms = now_ms - self._state.last_user_final_ms
                logger.info(
                    "Timing: user->tts_start turn=%s latency_ms=%.1f",
                    self._state.turn_index,
                    delta_ms,
                )

        if isinstance(frame, TTSStoppedFrame) and self._state.last_tts_start_ms is not None:
            now_ms = time.perf_counter() * 1000.0
            duration_ms = now_ms - self._state.last_tts_start_ms
            logger.info(
                "Timing: tts_duration turn=%s duration_ms=%.1f",
                self._state.turn_index,
                duration_ms,
            )

        await self.push_frame(frame, direction)


class _LatencyState:
    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._turn_id = 0
        self._current: dict = {}

    def _now_ms(self) -> float:
        return time.perf_counter() * 1000.0

    def _maybe_start_turn(self) -> None:
        if not self._current:
            self._turn_id += 1
            self._current = {
                "turn": self._turn_id,
                "t_start_ms": self._now_ms(),
            }

    def record(self, key: str, value) -> None:
        self._maybe_start_turn()
        if key not in self._current:
            self._current[key] = value

    def finalize(self) -> None:
        if not self._current:
            return
        self._current["t_logged_ms"] = self._now_ms()
        line = json.dumps(self._current, separators=(",", ":"))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._current = {}


class _LatencyUserProbe(FrameProcessor):
    def __init__(self, state: _LatencyState) -> None:
        super().__init__(name="LatencyUserProbe")
        self._state = state

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._state.record("speech_start_ms", self._state._now_ms())

        if isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            self._state.record("speech_end_ms", self._state._now_ms())

        if isinstance(frame, TranscriptionFrame) and frame.finalized:
            self._state.record("stt_final_ms", self._state._now_ms())
            self._state.record("stt_text", frame.text)

        await self.push_frame(frame, direction)


class _LatencyLLMProbe(FrameProcessor):
    def __init__(self, state: _LatencyState) -> None:
        super().__init__(name="LatencyLLMProbe")
        self._state = state

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame):
            self._state.record("llm_first_token_ms", self._state._now_ms())

        await self.push_frame(frame, direction)


class _LatencyTTSProbe(FrameProcessor):
    def __init__(self, state: _LatencyState) -> None:
        super().__init__(name="LatencyTTSProbe")
        self._state = state

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            self._state.record("tts_start_ms", self._state._now_ms())

        if isinstance(frame, TTSStoppedFrame):
            self._state.record("tts_end_ms", self._state._now_ms())
            self._state.finalize()

        await self.push_frame(frame, direction)

async def run_bot(room_url: str, token: str) -> None:
    logger.info("Starting bot — room: %s", room_url)

    log_dir = Path(os.getenv("LOG_DIR", "logs")).resolve()
    latency_log_path = log_dir / "latency.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)
    latency_log_path.touch(exist_ok=True)
    logger.info("Latency log path: %s", latency_log_path)

    # ── 1. LiveKit Transport ──────────────────────────────────────────────────
    transport = LiveKitTransport(
        url=room_url,
        token=token,
        room_name=os.getenv("ROOM_NAME", "plumbing-support"),
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.6)
            ),
            vad_audio_passthrough=True,
        ),
    )

    # ── 2. STT – Groq Whisper ─────────────────────────────────────────────────
    stt = GroqSTTService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="whisper-large-v3",
    )

    # ── 3. LLM – Groq (Llama 3) ──────────────────────────────────────────────
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=150,
        tools=TOOLS,
    )
    register_tools(llm)

    # ── 4. TTS – Cartesia streaming ───────────────────────────────────────────
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=os.getenv(
            "CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"
        ),
        model="sonic-english",
    )

    # ── 5. Conversation context ───────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages, TOOLS)
    context_aggregator = llm.create_context_aggregator(context)

    # ── 6. Pipeline assembly ──────────────────────────────────────────────────
    timing_state = _TimingState()
    latency_state = _LatencyState(latency_log_path)
    latency_user = _LatencyUserProbe(latency_state)
    latency_llm = _LatencyLLMProbe(latency_state)
    latency_tts = _LatencyTTSProbe(latency_state)
    user_timing = _UserTimingProbe(timing_state)
    bot_timing = _BotTimingProbe(timing_state)
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            latency_user,
            user_timing,
            context_aggregator.user(),
            llm,
            latency_llm,
            tts,
            latency_tts,
            bot_timing,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # ── 7. Task (interruptions enabled) ──────────────────────────────────────
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # ── 8. Greet the user as soon as they connect ─────────────────────────────
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport_obj, participant):
        logger.info("Participant joined — triggering greeting")
        # Push a "wake-up" user message so the LLM immediately generates
        # its opening greeting without waiting for the user to speak first.
        context.add_messages(
            [{"role": "user", "content": "Hello, I need help with a plumbing issue."}]
        )
        await task.queue_frames([OpenAILLMContextFrame(context)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport_obj, participant, reason):
        logger.info("Participant left (%s) — shutting down", reason)
        await task.cancel()

    # ── 9. Run ────────────────────────────────────────────────────────────────
    runner = PipelineRunner()
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Bot session ended cleanly.")
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AquaBot voice agent")
    parser.add_argument("--room-url", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    asyncio.run(run_bot(args.room_url, args.token))
