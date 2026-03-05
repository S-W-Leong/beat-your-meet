"""Beat Your Meet — AI Meeting Facilitator Agent.

This agent joins a LiveKit room, transcribes all participants,
and uses the LLM to naturally facilitate conversation against
the agenda. Time warnings and transitions are driven by asyncio
timers rather than a polling loop.
"""

import asyncio
import errno
import json
import logging
import os
import re
import socket
import sys
import time
from datetime import datetime
from dataclasses import dataclass as _dataclass
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobRequest, RunContext, WorkerOptions, cli, function_tool, llm as lk_llm
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, elevenlabs, openai, silero

from monitor import MeetingState, ItemState, ItemNotes, AgendaItem
from prompts import (
    BOT_INTRO_TEMPLATE,
    CHATTING_INTRO_TEMPLATE,
    CHATTING_SYSTEM_PROMPT,
    AGENDA_TRANSITION_TEMPLATE,
    FACILITATOR_SYSTEM_PROMPT,
    ITEM_SUMMARY_TOOL,
    ITEM_SUMMARY_PROMPT,
    STYLE_INSTRUCTIONS,
    TIME_WARNING_TEMPLATE,
)
from multi_audio import MultiParticipantAudioInput

# Resolve absolute path to .env so it works regardless of CWD or how Python
# was invoked (e.g. `python main.py` vs running from the project root).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("beat-your-meet")
logger.setLevel(logging.INFO)


def _coerce_llm_content_to_text(content):
    """Normalize streamed LLM delta content to text for LiveKit parser compatibility.

    Some providers may emit content as structured lists/parts instead of plain
    strings. LiveKit 1.4.3 expects a string in strip_thinking_tokens().
    """
    if content is None or isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = _coerce_llm_content_to_text(part)
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts) if parts else None

    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return _coerce_llm_content_to_text(content[key])
        return None

    for attr in ("text", "content", "value"):
        if hasattr(content, attr):
            return _coerce_llm_content_to_text(getattr(content, attr))

    return None


def _patch_livekit_thinking_strip() -> None:
    """Patch LiveKit thinking-token strip to handle non-string content safely."""
    from livekit.agents.llm import utils as _lk_llm_utils

    if getattr(_lk_llm_utils.strip_thinking_tokens, "_beat_safe_patched", False):
        return

    _orig = _lk_llm_utils.strip_thinking_tokens

    def _safe_strip_thinking_tokens(content, thinking):
        normalized = _coerce_llm_content_to_text(content)
        return _orig(normalized, thinking)

    _safe_strip_thinking_tokens._beat_safe_patched = True
    _lk_llm_utils.strip_thinking_tokens = _safe_strip_thinking_tokens


_patch_livekit_thinking_strip()

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}
_TIME_QUERY_PATTERNS = (
    re.compile(r"\bwhat(?:'s| is)?\s+time\b"),
    re.compile(r"\bwhat(?:'s| is)?\s+the\s+time\b"),
    re.compile(r"\bwhat\s+time\s+is\s+it\b"),
    re.compile(r"\bhow\s+long\s+has\s+this\s+meeting\b"),
    re.compile(r"\bhow\s+long\s+have\s+we\s+been\b"),
    re.compile(r"\bhow\s+much\s+time(?:\s+is)?\s+left\b"),
    re.compile(r"\btime\s+left\b"),
    re.compile(r"\bremaining\s+time\b"),
    re.compile(r"\bminutes?\s+left\b"),
)

_SKIP_PATTERNS = (
    re.compile(r"\bskip\s+(this|that|the\s+\w+|current)\b"),
    re.compile(r"\blet'?s?\s+skip\b"),
    re.compile(r"\bcan\s+we\s+skip\b"),
    re.compile(r"\bmove\s+on\s+to\s+the\s+next\b"),
    re.compile(r"\bnext\s+agenda\s+item\b"),
    re.compile(r"\bnext\s+topic\b"),
    re.compile(r"\bskip\s+ahead\b"),
)

_END_MEETING_PATTERNS = (
    re.compile(r"\bend\s+the\s+meeting\b"),
    re.compile(r"\bmeeting\s+is\s+(over|done|ended|finished)\b"),
    re.compile(r"\bmeeting'?s\s+(over|done|ended)\b"),
    re.compile(r"\blet'?s?\s+end\s+(the\s+)?meeting\b"),
    re.compile(r"\badjourn\b"),
    re.compile(r"\bthat'?s?\s+(it|all)\s+for\s+today\b"),
    re.compile(r"\bclose\s+(the\s+)?meeting\b"),
    re.compile(r"\bwe'?re?\s+done\s+(with\s+the\s+)?meeting\b"),
)

_BEAT_NAME_PATTERNS = (
    re.compile(r"\b(?:hey\s+)?beat\b"),
    re.compile(r"\bbeat[,!?]\b"),
    re.compile(r"^beat\b"),
    re.compile(r"\b@beat\b"),
)

_SILENCE_PHRASES = [
    "please be quiet", "be quiet", "quiet please", "stop talking",
    "stop interrupting", "don't interrupt", "we've got this", "we're fine",
    "let us talk", "hold on bot", "hold on beat", "not now", "stay quiet",
    "zip it", "shh", "shut up", "shut it", "pipe down", "hush",
    "silence beat", "stop talking beat", "beat stop", "beat be quiet",
    "beat shut up",
]

_OVERRIDE_PATTERNS = (
    re.compile(r"\bkeep\s+going\b"),
    re.compile(r"\blet'?s?\s+continue\b"),
    re.compile(r"\bwe'?re?\s+not\s+done\b"),
    re.compile(r"\bmore\s+time\b"),
    re.compile(r"\bextend\b"),
)


@_dataclass
class DocumentRequest:
    doc_type: str  # "attendance" | "action_items" | "custom"
    description: str  # used as LLM prompt hint for "custom"
    slug: str  # filename stem e.g. "attendance", "concerns"


_DOC_REQUEST_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    (
        re.compile(r"\b(take|do|track|record)\s+(an?\s+)?attendance\b"),
        "attendance",
        "Record which participants attended the meeting",
        "attendance",
    ),
    (
        re.compile(r"\bheadcount\b"),
        "attendance",
        "Record which participants attended the meeting",
        "attendance",
    ),
    (
        re.compile(
            r"\b(list|track|note|collect)\s+(the\s+)?(all\s+)?action\s+items?\b"
        ),
        "action_items",
        "Consolidated list of all action items from the meeting",
        "action-items",
    ),
    (
        re.compile(r"\b(make|write|create|prepare|generate)\s+(a\s+)?summary\b"),
        "summary",
        "Full meeting summary with key points and decisions",
        "summary",
    ),
]

_CUSTOM_DOC_PATTERN = re.compile(
    r"\b(keep\s+(a\s+)?(record|log|track|note)\s+of|"
    r"document|note\s+down|record\s+down)\b"
)


def _bg_task(coro, *, name: str) -> asyncio.Task:
    """Create a fire-and-forget task that logs exceptions instead of discarding them."""
    task = asyncio.create_task(coro, name=name)

    def _on_done(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception() is not None:
            logger.error("Background task %r failed", name, exc_info=t.exception())

    task.add_done_callback(_on_done)
    return task


def _resolve_agent_port() -> int:
    """Resolve agent health-check port from AGENT_PORT with sane defaults."""
    raw = os.environ.get("AGENT_PORT", "8081")
    try:
        port = int(raw)
    except ValueError:
        logger.warning("Invalid AGENT_PORT=%r; defaulting to 8081", raw)
        return 8081

    if not (0 <= port <= 65535):
        logger.warning("Out-of-range AGENT_PORT=%r; defaulting to 8081", raw)
        return 8081
    return port


def _is_port_in_use(port: int) -> bool:
    """Return True when a TCP port is already bound on localhost."""
    if port == 0:
        return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            return False
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return True
            raise


def _resolve_bool_env(name: str, default: bool) -> bool:
    """Resolve a bool env var from common truthy/falsey strings."""
    raw = os.environ.get(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSEY:
        return False

    logger.warning(
        "Invalid %s=%r; expected one of %s or %s. Defaulting to %s.",
        name, raw, sorted(_TRUTHY), sorted(_FALSEY), default,
    )
    return default


def _extract_latest_user_text(chat_ctx: lk_llm.ChatContext) -> str:
    """Extract the most recent user text message from chat context."""
    for item in reversed(chat_ctx.items):
        if getattr(item, "type", None) != "message":
            continue
        if getattr(item, "role", None) != "user":
            continue
        text = getattr(item, "text_content", None)
        if text:
            return text
    return ""


def _is_time_query(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _TIME_QUERY_PATTERNS)


def _is_skip_request(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _SKIP_PATTERNS)


def _is_end_meeting_request(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _END_MEETING_PATTERNS)


def _is_addressed_to_beat(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _BEAT_NAME_PATTERNS)


def _detect_silence_request(text: str) -> bool:
    lowered = text.strip().lower()
    return any(phrase in lowered for phrase in _SILENCE_PHRASES)


def _is_override_request(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _OVERRIDE_PATTERNS)


def _detect_doc_request(text: str) -> DocumentRequest | None:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return None
    for pattern, doc_type, description, slug in _DOC_REQUEST_PATTERNS:
        if pattern.search(normalized):
            return DocumentRequest(doc_type=doc_type, description=description, slug=slug)
    if _CUSTOM_DOC_PATTERN.search(normalized):
        slug = re.sub(r"[^a-z0-9]+", "-", normalized[:40]).strip("-")
        return DocumentRequest(
            doc_type="custom",
            description=text.strip(),
            slug=slug or "custom",
        )
    return None


def _format_duration_for_tts(minutes: float) -> str:
    total_seconds = max(0, int(round(minutes * 60)))
    whole_minutes, seconds = divmod(total_seconds, 60)

    if whole_minutes and seconds:
        minute_word = "minute" if whole_minutes == 1 else "minutes"
        second_word = "second" if seconds == 1 else "seconds"
        return f"{whole_minutes} {minute_word} {seconds} {second_word}"
    if whole_minutes:
        minute_word = "minute" if whole_minutes == 1 else "minutes"
        return f"{whole_minutes} {minute_word}"
    second_word = "second" if seconds == 1 else "seconds"
    return f"{seconds} {second_word}"


def _format_time_status_for_tts(status: dict) -> str:
    if not status.get("meeting_started", False):
        return "The meeting clock has not started yet."

    clock_value = "unknown time"
    current_time_iso = status.get("current_time_iso")
    if isinstance(current_time_iso, str) and current_time_iso:
        try:
            dt = datetime.fromisoformat(current_time_iso)
            clock_value = dt.strftime("%I:%M %p").lstrip("0")
        except ValueError:
            clock_value = current_time_iso

    total_elapsed = _format_duration_for_tts(
        float(status.get("total_meeting_minutes", 0.0))
    )
    remaining = _format_duration_for_tts(
        float(status.get("current_item_remaining_minutes", 0.0))
    )
    topic = str(status.get("current_item_topic") or "the current item")
    allocated = float(status.get("current_item_allocated_minutes", 0.0))

    if topic == "none" or allocated <= 0:
        return f"It's {clock_value}. The meeting has run for {total_elapsed}, and there is no active agenda item right now."

    return (
        f"It's {clock_value}. The meeting has run for {total_elapsed}, "
        f"with {remaining} left on {topic}."
    )


# ---------------------------------------------------------------------------
# AgendaTimers — event-driven time management
# ---------------------------------------------------------------------------


class AgendaTimers:
    """Manages asyncio timers for agenda item warnings and transitions.

    Replaces the polling-based monitoring loop with precise one-shot timers.
    """

    def __init__(
        self,
        session: AgentSession,
        state: MeetingState,
        room: rtc.Room,
        agent: Agent,
        mistral_client: "Mistral",
    ) -> None:
        self._session = session
        self._state = state
        self._room = room
        self._agent = agent
        self._mistral_client = mistral_client
        self._warning_handle: asyncio.TimerHandle | None = None
        self._overtime_handle: asyncio.TimerHandle | None = None
        self._heartbeat_task: asyncio.Task | None = None

    def start_item_timers(self) -> None:
        """Set timers for the current agenda item's warning and overtime."""
        self.cancel()
        item = self._state.current_item
        if item is None or self._state.style == "chatting":
            return

        duration_seconds = item.duration_minutes * 60
        loop = asyncio.get_event_loop()

        # Warning at 80% of allocated time
        warning_delay = duration_seconds * 0.8
        if warning_delay > 0:
            self._warning_handle = loop.call_later(
                warning_delay,
                lambda: _bg_task(self._on_warning(), name="timer_warning"),
            )

        # Overtime at 100% of allocated time
        if duration_seconds > 0:
            self._overtime_handle = loop.call_later(
                duration_seconds,
                lambda: _bg_task(self._on_overtime(), name="timer_overtime"),
            )

        logger.info(
            "Timers set for '%s': warning=%.0fs, overtime=%.0fs",
            item.topic, warning_delay, duration_seconds,
        )

    def extend(self, grace_seconds: float = 120.0) -> None:
        """Cancel the overtime timer and reschedule it after a grace period."""
        if self._overtime_handle:
            self._overtime_handle.cancel()
            self._overtime_handle = None
        loop = asyncio.get_event_loop()
        self._overtime_handle = loop.call_later(
            grace_seconds,
            lambda: _bg_task(self._on_overtime(), name="timer_overtime_extended"),
        )
        logger.info("Override: overtime timer extended by %.0fs", grace_seconds)

    def cancel(self) -> None:
        """Cancel all pending timers for the current item."""
        if self._warning_handle:
            self._warning_handle.cancel()
            self._warning_handle = None
        if self._overtime_handle:
            self._overtime_handle.cancel()
            self._overtime_handle = None

    def start_heartbeat(self) -> None:
        """Start a 60s heartbeat that sends agenda state to the frontend."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            return
        self._heartbeat_task = _bg_task(self._heartbeat_loop(), name="heartbeat")

    def stop(self) -> None:
        """Cancel all timers and stop the heartbeat."""
        self.cancel()
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

    async def _on_warning(self) -> None:
        """Fired at 80% of the item's allocated time."""
        item = self._state.current_item
        if item is None or item.state == ItemState.COMPLETED:
            return

        item.state = ItemState.WARNING
        remaining = max(0, item.duration_minutes - self._state.elapsed_minutes)
        warning = TIME_WARNING_TEMPLATE.format(
            remaining=f"{remaining:.0f}",
            topic=item.topic,
        )

        if self._state.can_intervene():
            try:
                await self._session.say(warning, allow_interruptions=True)
                self._state.record_intervention()
                logger.info("Time warning delivered for '%s'", item.topic)
            except RuntimeError:
                logger.warning("Session closed — skipping time warning")
                return

        await _send_agenda_state(self._room, self._state)

    async def _on_overtime(self) -> None:
        """Fired at 100% of the item's allocated time — auto-advance."""
        item = self._state.current_item
        if item is None or item.state == ItemState.COMPLETED:
            return

        item.state = ItemState.OVERTIME
        prev_index = self._state.current_item_index
        completed_item = self._state.current_item

        next_item = self._state.advance_to_next()

        try:
            if next_item:
                transition = AGENDA_TRANSITION_TEMPLATE.format(
                    next_item=next_item.topic,
                    duration=int(next_item.duration_minutes),
                )
                await self._session.say(transition, allow_interruptions=True)
                self._state.record_intervention()
                # Start timers for the new item
                self.start_item_timers()
            else:
                # No more items — wait for explicit host-driven UI end signal.
                await self._session.say(
                    "That wraps up our agenda. Host, please click End Meeting when you're ready.",
                    allow_interruptions=True,
                )
                self._state.record_intervention()
        except RuntimeError:
            logger.warning("Session closed — skipping overtime transition")
            return

        # Summarise the completed item
        if completed_item:
            transcript = self._state.get_item_transcript(prev_index)
            notes = await _summarize_item(self._mistral_client, completed_item, transcript)
            self._state.meeting_notes.append(notes)
            await _refresh_agent_instructions(self._agent, self._state)

        await _send_agenda_state(self._room, self._state)

    async def _heartbeat_loop(self) -> None:
        """Send agenda state to the frontend every 60s for clock drift correction."""
        while not self._state.meeting_end_triggered:
            await asyncio.sleep(60)
            if not self._state.meeting_end_triggered:
                await _send_agenda_state(self._room, self._state)


# ---------------------------------------------------------------------------
# BeatFacilitatorAgent
# ---------------------------------------------------------------------------


class BeatFacilitatorAgent(Agent):
    """Agent that handles deterministic commands (time, skip, end) without
    LLM calls and delegates everything else to the main LLM."""

    def __init__(
        self,
        *,
        meeting_state: MeetingState,
        deterministic_time_queries_enabled: bool,
        room: "rtc.Room | None" = None,
        timers: "AgendaTimers | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._meeting_state = meeting_state
        self._deterministic_time_queries_enabled = deterministic_time_queries_enabled
        self._room = room
        self._timers = timers
        logger.info(
            "BeatFacilitatorAgent registered %d tools: %s",
            len(self.tools),
            [t.id for t in self.tools],
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        latest_user_text = _extract_latest_user_text(chat_ctx)
        is_addressed_to_beat = _is_addressed_to_beat(latest_user_text)

        # Silence mode: only respond when explicitly addressed
        if self._meeting_state.is_silenced and not is_addressed_to_beat:
            remaining = max(0, self._meeting_state.silence_requested_until - time.time())
            logger.info("silence_mode: suppressing response (%.0fs remaining)", remaining)
            return

        # Deterministic time query
        if _is_time_query(latest_user_text) and self._deterministic_time_queries_enabled:
            status = self._meeting_state.get_time_status()
            logger.info("time_query: deterministic response")
            yield _format_time_status_for_tts(status)
            return

        # Skip current agenda item
        if _is_skip_request(latest_user_text) and self._room:
            state = self._meeting_state
            current_topic = state.current_item.topic if state.current_item else None
            if current_topic:
                if self._timers:
                    self._timers.cancel()
                next_item = state.advance_to_next()
                _bg_task(_send_agenda_state(self._room, state), name="send_agenda_state_skip")
                _bg_task(_refresh_agent_instructions(self, state), name="refresh_instructions_skip")
                state.record_intervention()
                if next_item:
                    if self._timers:
                        self._timers.start_item_timers()
                    logger.info("skip: '%s' → '%s'", current_topic, next_item.topic)
                    yield f"Sure, skipping {current_topic}. Moving on to {next_item.topic}."
                else:
                    logger.info("skip: '%s' was the last item", current_topic)
                    yield f"Sure, skipping {current_topic}. That was the last agenda item — great meeting everyone!"
                return

        # End meeting requests are UI-only (host presses End Meeting button).
        if _is_end_meeting_request(latest_user_text):
            yield "Please use the End Meeting button in the UI to end this meeting."
            return

        # Override ("keep going") — extend the overtime timer
        if _is_override_request(latest_user_text) and self._timers:
            self._timers.extend()
            yield "Got it, I'll give you more time on this one!"
            return

        # Document request
        doc_req = _detect_doc_request(latest_user_text)
        if doc_req and self._room:
            existing_slugs = {r.slug for r in self._meeting_state.doc_requests}
            if doc_req.slug not in existing_slugs:
                self._meeting_state.doc_requests.append(doc_req)
                logger.info("doc_request queued: type=%s slug=%s", doc_req.doc_type, doc_req.slug)
            yield "Got it — I'll prepare that document at the end of the meeting."
            return

        # Default: let the LLM handle it
        logger.info("llm_node: calling LLM addressed_to_beat=%s text=%r", is_addressed_to_beat, latest_user_text[:80])
        try:
            async for chunk in super().llm_node(chat_ctx, tools, model_settings):
                yield chunk
        except Exception:
            logger.exception("llm_node: LLM call failed for utterance %r", latest_user_text[:80])
            if is_addressed_to_beat:
                yield "Sorry, I had a technical hiccup — could you repeat that?"

    # ------------------------------------------------------------------
    # Function tools — auto-discovered by LiveKit and exposed to the LLM
    # ------------------------------------------------------------------

    @function_tool()
    async def get_participant_count(self, context: RunContext) -> dict:
        """Get the current number of participants in the meeting room.

        Use this when someone asks how many people are in the meeting,
        who is here, or anything about attendance.
        """
        if not self._room:
            return {"participant_count": 0, "participants": []}
        participants = list(self._room.remote_participants.values())
        return {
            "participant_count": len(participants),
            "participants": [p.identity for p in participants],
        }

    @function_tool()
    async def get_meeting_info(self, context: RunContext) -> dict:
        """Get current meeting status including timing and progress.

        Use this when someone asks about meeting progress, how long the meeting
        has been running, what topic is being discussed, or the meeting style.
        """
        state = self._meeting_state
        item = state.current_item
        return {
            "agenda_title": state.agenda_title,
            "style": state.style,
            "current_item": item.topic if item else None,
            "current_item_description": item.description if item else None,
            "current_item_elapsed_minutes": round(state.elapsed_minutes, 1),
            "current_item_allocated_minutes": item.duration_minutes if item else 0,
            "total_meeting_minutes": round(state.total_meeting_minutes, 1),
            "total_scheduled_minutes": round(state.total_scheduled_minutes, 1),
            "meeting_overtime_minutes": round(state.meeting_overtime, 1),
            "items_completed": sum(1 for i in state.items if i.state == ItemState.COMPLETED),
            "items_remaining": len(state.remaining_items),
            "total_items": len(state.items),
        }

    @function_tool()
    async def get_agenda(self, context: RunContext) -> dict:
        """Get the full meeting agenda with all items and their current status.

        Use this when someone asks what's on the agenda, what topics are planned,
        or wants an overview of the meeting structure.
        """
        state = self._meeting_state
        return {
            "title": state.agenda_title,
            "items": [
                {
                    "id": item.id,
                    "topic": item.topic,
                    "description": item.description,
                    "duration_minutes": item.duration_minutes,
                    "state": item.state.value,
                    "actual_elapsed_minutes": round(item.actual_elapsed, 1),
                }
                for item in state.items
            ],
        }

    @function_tool()
    async def get_meeting_notes(self, context: RunContext) -> dict:
        """Get notes and summaries from completed agenda items.

        Use this when someone asks for a recap, summary, what was discussed,
        decisions made, or action items from earlier in the meeting.
        """
        state = self._meeting_state
        return {
            "notes": [
                {
                    "topic": n.topic,
                    "key_points": n.key_points,
                    "decisions": n.decisions,
                    "action_items": n.action_items,
                }
                for n in state.meeting_notes
            ],
        }

    @function_tool()
    async def web_search(self, context: RunContext, query: str) -> dict:
        """Search the web for current information, latest news, facts, or data.

        Use this when a participant asks about recent events, news, statistics,
        facts you're unsure about, or anything that requires up-to-date information
        beyond your training data. Keep queries concise and specific.

        Args:
            query: The search query string.
        """
        logger.info("web_search: searching for %r", query)
        try:
            # Avoid runtime issues from third-party DDG clients by using
            # DuckDuckGo's HTML endpoint directly via stdlib networking.
            def _search_sync() -> list[dict]:
                import html
                from urllib.parse import parse_qs, quote_plus, unquote, urlparse
                from urllib.request import Request, urlopen

                result_link_re = re.compile(
                    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    re.IGNORECASE | re.DOTALL,
                )
                snippet_re = re.compile(
                    r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                    re.IGNORECASE | re.DOTALL,
                )

                def _clean(s: str) -> str:
                    s = re.sub(r"<[^>]+>", " ", s)
                    s = html.unescape(s)
                    return re.sub(r"\s+", " ", s).strip()

                def _unwrap_ddg_url(href: str) -> str:
                    if href.startswith("//"):
                        href = f"https:{href}"
                    parsed = urlparse(href)
                    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
                        uddg = parse_qs(parsed.query).get("uddg", [None])[0]
                        if uddg:
                            return unquote(uddg)
                    return href

                search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
                req = Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=10) as resp:
                    page = resp.read().decode("utf-8", errors="ignore")

                links = result_link_re.findall(page)
                snippets = snippet_re.findall(page)
                results: list[dict] = []
                for i, (href, title_html) in enumerate(links[:5]):
                    snippet_html = snippets[i] if i < len(snippets) else ""
                    results.append(
                        {
                            "title": _clean(title_html),
                            "snippet": _clean(snippet_html),
                            "url": _unwrap_ddg_url(href),
                        }
                    )
                return results

            results = await asyncio.to_thread(_search_sync)
            if not results:
                return {"results": [], "message": "No results found."}
            return {"results": results}
        except Exception as e:
            logger.exception("web_search failed for query %r", query)
            return {"results": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def entrypoint(ctx: JobContext):
    try:
        logger.info("Explicit dispatch accepted — joining room %s", ctx.room.name)
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Wait for the first human participant to join
        logger.info("Waiting for participant...")
        participant = await ctx.wait_for_participant()
        logger.info(f"Participant joined: {participant.identity}")

        # Parse room metadata for agenda and style
        logger.info("Parsing room metadata...")
        room_metadata = ctx.room.metadata
        if room_metadata:
            metadata = json.loads(room_metadata)
        else:
            metadata = {
                "agenda": {
                    "title": "Meeting",
                    "items": [
                        {
                            "id": 1,
                            "topic": "Open Discussion",
                            "description": "General discussion",
                            "duration_minutes": 30,
                        }
                    ],
                },
                "style": "moderate",
            }

        expected_host_token = metadata.get("host_token")

        # Initialize meeting state
        meeting_state = MeetingState.from_metadata(metadata)
        logger.info(
            f"Meeting state initialized: {len(meeting_state.items)} items, style={meeting_state.style}"
        )

        # Configure LLM (Mistral via OpenAI-compatible interface)
        mistral_llm = openai.LLM(
            model="mistral-large-latest",
            api_key=os.environ["MISTRAL_API_KEY"],
            base_url="https://api.mistral.ai/v1",
        )

        # Shared Mistral client for item summaries and chat @beat responses
        from mistralai import Mistral as _Mistral
        mistral_chat_client = _Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        # Build system instructions
        if meeting_state.style == "chatting":
            instructions = CHATTING_SYSTEM_PROMPT
        else:
            ctx_data = meeting_state.get_context_for_prompt()
            instructions = FACILITATOR_SYSTEM_PROMPT.format(
                style_instructions=STYLE_INSTRUCTIONS.get(
                    meeting_state.style, STYLE_INSTRUCTIONS["moderate"]
                ),
                **ctx_data,
            )

        deterministic_time_queries_enabled = _resolve_bool_env(
            "DETERMINISTIC_TIME_QUERIES", default=True
        )

        # Create the voice agent and session
        logger.info("Creating voice agent (VAD + STT + LLM + TTS)...")
        agent = BeatFacilitatorAgent(
            instructions=instructions,
            vad=silero.VAD.load(),
            stt=deepgram.STT(model="nova-2", language="en"),
            llm=mistral_llm,
            tts=elevenlabs.TTS(model="eleven_turbo_v2_5"),
            meeting_state=meeting_state,
            deterministic_time_queries_enabled=deterministic_time_queries_enabled,
            room=ctx.room,
        )

        session = AgentSession()

        # Create the timer system
        timers = AgendaTimers(
            session=session,
            state=meeting_state,
            room=ctx.room,
            agent=agent,
            mistral_client=mistral_chat_client,
        )
        agent._timers = timers

        # Track transcriptions
        @session.on("user_input_transcribed")
        def on_speech(event):
            if not event.is_final:
                return
            remote = list(ctx.room.remote_participants.values())
            speaker = remote[0].identity if len(remote) == 1 else "participant"
            meeting_state.add_transcript(speaker, event.transcript)
            if _detect_silence_request(event.transcript):
                meeting_state.update_silence_signal()
                logger.info("[silence] Silence signal set by %s", speaker)
            logger.info(f"[{speaker}] {event.transcript}")

        @ctx.room.on("data_received")
        def on_data_received(data: rtc.DataPacket):
            try:
                msg = json.loads(data.data.decode())

                # Style change from any participant
                if (
                    msg.get("type") == "set_style"
                    and msg.get("style") in ("gentle", "moderate", "chatting")
                ):
                    meeting_state.style = msg["style"]
                    logger.info(f"Style changed to {msg['style']}")
                    _bg_task(_refresh_agent_instructions(agent, meeting_state), name="refresh_instructions_style")
                    # Restart timers — chatting mode disables them
                    if msg["style"] == "chatting":
                        timers.cancel()
                    else:
                        timers.start_item_timers()

                # End meeting triggered from UI
                elif msg.get("type") == "end_meeting":
                    provided_host_token = msg.get("host_token")
                    if expected_host_token and provided_host_token != expected_host_token:
                        logger.warning("Ignoring end_meeting with invalid host token")
                        return
                    logger.info("end_meeting signal received from host UI")
                    timers.stop()
                    _bg_task(
                        _end_meeting(ctx.room, meeting_state, mistral_client=mistral_chat_client),
                        name="end_meeting_ui",
                    )

                # @beat mention in the chat
                elif msg.get("type") == "chat_message" and not msg.get("is_agent"):
                    text = msg.get("text", "")
                    if text.strip().lower().startswith("@beat"):
                        question = text.strip()[5:].strip()
                        sender = msg.get("sender", "someone")
                        _bg_task(
                            _handle_chat_mention(
                                ctx.room, meeting_state, mistral_chat_client, agent, sender, question
                            ),
                            name="chat_mention",
                        )
            except Exception:
                logger.exception("Failed to handle data_received")

        # Use a custom audio input that mixes ALL participants' audio
        multi_audio = MultiParticipantAudioInput(ctx.room)
        session.input.audio = multi_audio

        # Start the session
        logger.info("Starting agent session...")
        await session.start(agent, room=ctx.room)

        # Start the meeting
        meeting_state.start_meeting()
        await _refresh_agent_instructions(agent, meeting_state)

        # Start timers for the first agenda item
        if meeting_state.style != "chatting":
            timers.start_item_timers()
        timers.start_heartbeat()

        # Deliver bot introduction
        if meeting_state.style == "chatting":
            intro = CHATTING_INTRO_TEMPLATE
        else:
            first_item = meeting_state.items[0] if meeting_state.items else None
            intro = BOT_INTRO_TEMPLATE.format(
                num_items=len(meeting_state.items),
                total_minutes=int(meeting_state.total_scheduled_minutes),
                first_item=first_item.topic if first_item else "the discussion",
            )
        await asyncio.sleep(2)
        await session.say(intro, allow_interruptions=True)
        meeting_state.record_intervention()

        # Send initial agenda state to frontend
        await _send_agenda_state(ctx.room, meeting_state)

        logger.info("Beat Your Meet agent started successfully")

    except Exception:
        logger.exception("Agent entrypoint crashed")
        raise


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


async def _summarize_item(
    client: "Mistral",
    item: AgendaItem,
    transcript: str,
) -> ItemNotes:
    """Summarise a completed agenda item using Mistral Small + tool calling."""
    try:
        response = await client.chat.complete_async(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": ITEM_SUMMARY_PROMPT.format(
                        topic=item.topic,
                        description=item.description,
                        transcript=transcript or "(no transcript recorded)",
                    ),
                }
            ],
            tools=[ITEM_SUMMARY_TOOL],
            tool_choice="any",
            temperature=0.2,
            max_tokens=512,
        )
        if response.choices[0].message.tool_calls:
            args = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            logger.info(f"Item summarization complete for '{item.topic}'")
            return ItemNotes(
                item_id=item.id,
                topic=item.topic,
                key_points=args.get("key_points", []),
                decisions=args.get("decisions", []),
                action_items=args.get("action_items", []),
            )
    except Exception as e:
        logger.error(f"Item summarization failed: {e}")
    return ItemNotes(item_id=item.id, topic=item.topic)


async def _chat_web_search(query: str) -> dict:
    """Standalone web search for chat @beat mentions (reuses the DDG HTML approach)."""
    try:
        def _search_sync() -> list[dict]:
            import html
            from urllib.parse import parse_qs, quote_plus, unquote, urlparse
            from urllib.request import Request, urlopen

            result_link_re = re.compile(
                r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )
            snippet_re = re.compile(
                r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
                re.IGNORECASE | re.DOTALL,
            )

            def _clean(s: str) -> str:
                s = re.sub(r"<[^>]+>", " ", s)
                s = html.unescape(s)
                return re.sub(r"\s+", " ", s).strip()

            def _unwrap_ddg_url(href: str) -> str:
                if href.startswith("//"):
                    href = f"https:{href}"
                parsed = urlparse(href)
                if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
                    uddg = parse_qs(parsed.query).get("uddg", [None])[0]
                    if uddg:
                        return unquote(uddg)
                return href

            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            req = Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as resp:
                page = resp.read().decode("utf-8", errors="ignore")

            links = result_link_re.findall(page)
            snippets = snippet_re.findall(page)
            results: list[dict] = []
            for i, (href, title_html) in enumerate(links[:5]):
                snippet_html = snippets[i] if i < len(snippets) else ""
                results.append(
                    {
                        "title": _clean(title_html),
                        "snippet": _clean(snippet_html),
                        "url": _unwrap_ddg_url(href),
                    }
                )
            return results

        results = await asyncio.to_thread(_search_sync)
        if not results:
            return {"results": [], "message": "No results found."}
        return {"results": results}
    except Exception as e:
        logger.exception("_chat_web_search failed for query %r", query)
        return {"results": [], "error": str(e)}


async def _handle_chat_mention(
    room: rtc.Room,
    state: MeetingState,
    client: "Mistral",
    agent: Agent,
    sender: str,
    question: str,
):
    """Handle an @beat chat mention — mirrors all voice capabilities."""

    async def _reply(text: str) -> None:
        payload = json.dumps({
            "type": "chat_message",
            "sender": "Beat",
            "text": text,
            "is_agent": True,
            "timestamp": time.time(),
        }).encode()
        try:
            await room.local_participant.publish_data(payload, reliable=True, topic="chat")
        except Exception:
            logger.exception("Failed to publish @beat chat response")

    # Skip current agenda item
    if _is_skip_request(question):
        current_topic = state.current_item.topic if state.current_item else None
        if current_topic:
            next_item = state.advance_to_next()
            await _send_agenda_state(room, state)
            await _refresh_agent_instructions(agent, state)
            state.record_intervention()
            if next_item:
                await _reply(f"Skipping {current_topic}. Moving on to {next_item.topic}.")
            else:
                await _reply(f"Skipping {current_topic}. That was the last agenda item — great meeting!")
        else:
            await _reply("There are no active agenda items to skip.")
        return

    # End meeting requests are UI-only (host presses End Meeting button).
    if _is_end_meeting_request(question):
        await _reply("Please ask the host to use the End Meeting button in the UI.")
        return

    # Time query (deterministic)
    if _is_time_query(question):
        status = state.get_time_status()
        await _reply(_format_time_status_for_tts(status))
        return

    # Document request
    doc_req = _detect_doc_request(question)
    if doc_req:
        existing_slugs = {r.slug for r in state.doc_requests}
        if doc_req.slug not in existing_slugs:
            state.doc_requests.append(doc_req)
        await _reply("Got it — I'll prepare that document at the end of the meeting.")
        return

    # General question → Mistral with tool support
    item = state.current_item
    ctx_summary = (
        f"Meeting style: {state.style}. "
        f"Current agenda item: '{item.topic}', "
        f"elapsed {state.elapsed_minutes:.1f} of {item.duration_minutes} min."
        if item
        else f"Meeting style: {state.style}. No active agenda item."
    )

    # Tool definitions for chat @beat responses
    chat_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_participant_count",
                "description": "Get the current number of participants in the meeting room.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_meeting_info",
                "description": "Get current meeting status including timing and progress.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_agenda",
                "description": "Get the full meeting agenda with all items and their status.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_meeting_notes",
                "description": "Get notes and summaries from completed agenda items.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, news, facts, or data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query string."},
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    # Tool execution helpers
    async def _exec_tool(name: str, args: dict) -> str:
        if name == "get_participant_count":
            participants = list(room.remote_participants.values())
            result = {
                "participant_count": len(participants),
                "participants": [p.identity for p in participants],
            }
        elif name == "get_meeting_info":
            result = {
                "agenda_title": state.agenda_title,
                "style": state.style,
                "current_item": item.topic if item else None,
                "current_item_elapsed_minutes": round(state.elapsed_minutes, 1),
                "current_item_allocated_minutes": item.duration_minutes if item else 0,
                "total_meeting_minutes": round(state.total_meeting_minutes, 1),
                "items_completed": sum(1 for i in state.items if i.state == ItemState.COMPLETED),
                "items_remaining": len(state.remaining_items),
                "total_items": len(state.items),
            }
        elif name == "get_agenda":
            result = {
                "title": state.agenda_title,
                "items": [
                    {
                        "topic": i.topic,
                        "duration_minutes": i.duration_minutes,
                        "state": i.state.value,
                    }
                    for i in state.items
                ],
            }
        elif name == "get_meeting_notes":
            result = {
                "notes": [
                    {
                        "topic": n.topic,
                        "key_points": n.key_points,
                        "decisions": n.decisions,
                        "action_items": n.action_items,
                    }
                    for n in state.meeting_notes
                ],
            }
        elif name == "web_search":
            query = args.get("query", "")
            result = await _chat_web_search(query)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Beat, an AI meeting assistant replying in a text chat. "
                    f"{ctx_summary} "
                    "You have tools to look up meeting data and search the web. "
                    "USE your tools when the question requires real data (participant counts, "
                    "agenda info, web searches, etc.) instead of guessing. "
                    "Be concise and helpful. 1–3 sentences max. Plain text only."
                ),
            },
            {
                "role": "user",
                "content": question if question else "You were mentioned.",
            },
        ]

        # Tool-calling loop (max 3 rounds to prevent infinite loops)
        reply_text = "Sorry, I couldn't process that."
        response = None
        for _ in range(3):
            response = await client.chat.complete_async(
                model="mistral-small-latest",
                messages=messages,
                tools=chat_tools,
                temperature=0.4,
                max_tokens=300,
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" or (choice.message.tool_calls and len(choice.message.tool_calls) > 0):
                # Append assistant message with tool calls
                messages.append(choice.message)
                # Execute each tool call and append results
                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    logger.info("chat_mention: calling tool %s(%s)", fn_name, fn_args)
                    tool_result = await _exec_tool(fn_name, fn_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                continue  # Let the LLM process tool results

            # No tool calls — we have the final text response
            reply_text = (choice.message.content or "").strip()
            break
        else:
            # All 3 rounds used tool calls — use whatever we got last
            if response and response.choices:
                reply_text = (response.choices[0].message.content or "").strip() or reply_text

    except Exception:
        logger.exception("Chat @beat LLM call failed")
        reply_text = "Sorry, I couldn't process that right now."

    await _reply(reply_text)


async def _refresh_agent_instructions(
    agent: Agent,
    state: MeetingState,
):
    """Rebuild the facilitator system prompt and push it to the agent."""
    if state.style == "chatting":
        await agent.update_instructions(CHATTING_SYSTEM_PROMPT)
    else:
        ctx_data = state.get_context_for_prompt()
        new_instructions = FACILITATOR_SYSTEM_PROMPT.format(
            style_instructions=STYLE_INSTRUCTIONS.get(
                state.style, STYLE_INSTRUCTIONS["moderate"]
            ),
            **ctx_data,
        )
        await agent.update_instructions(new_instructions)
    logger.info(f"Agent instructions refreshed (style={state.style})")


async def _end_meeting(
    room: rtc.Room,
    state: MeetingState,
    *,
    mistral_client: "Mistral | None" = None,
) -> None:
    """Publish end-of-meeting signals and generate/upload documents."""
    if state.meeting_end_triggered:
        return
    state.meeting_end_triggered = True

    try:
        payload = json.dumps({"type": "meeting_ended"}).encode()
        await room.local_participant.publish_data(payload, reliable=True, topic="agenda")
    except Exception:
        logger.exception("Failed to publish meeting_ended")

    from doc_generator import generate_and_upload_all_docs

    room_id = room.name
    server_url = os.environ.get("SERVER_URL", "http://localhost:8000")
    try:
        if mistral_client is None:
            from mistralai import Mistral as _Mistral
            mistral_client = _Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        await generate_and_upload_all_docs(
            mistral_client=mistral_client,
            state=state,
            room_id=room_id,
            server_url=server_url,
        )
        payload = json.dumps({"type": "docs_ready", "room_id": room_id}).encode()
        await room.local_participant.publish_data(payload, reliable=True, topic="agenda")
        logger.info("docs_ready signal sent for room %s", room_id)
    except Exception:
        logger.exception("Document generation/upload failed")


async def _send_agenda_state(room: rtc.Room, state: MeetingState):
    """Send current agenda state to frontend via data channel."""
    now_epoch = time.time()
    payload = json.dumps(
        {
            "type": "agenda_state",
            "current_item_index": state.current_item_index,
            "items": [
                {
                    "id": item.id,
                    "topic": item.topic,
                    "duration_minutes": item.duration_minutes,
                    "state": item.state.value,
                    "actual_elapsed": item.actual_elapsed,
                }
                for item in state.items
            ],
            "elapsed_minutes": state.elapsed_minutes,
            "meeting_overtime": state.meeting_overtime,
            "total_meeting_minutes": state.total_meeting_minutes,
            "style": state.style,
            "server_now_epoch": now_epoch,
            "meeting_start_epoch": state.meeting_start_time,
            "item_start_epoch": state.item_start_time,
            "meeting_notes": [
                {
                    "item_id": n.item_id,
                    "topic": n.topic,
                    "key_points": n.key_points,
                    "decisions": n.decisions,
                    "action_items": n.action_items,
                }
                for n in state.meeting_notes
            ],
        }
    ).encode()

    try:
        await room.local_participant.publish_data(
            payload, reliable=True, topic="agenda"
        )
    except Exception as e:
        logger.warning(f"Failed to send agenda state: {e}")


async def request_fnc(req: JobRequest) -> None:
    """Only accept the job if the room's metadata has invite_bot=True."""
    metadata_str = req.room.metadata
    logger.info(
        "request_fnc called — room=%s, agent_name=%s, metadata=%s",
        req.room.name, req.agent_name, metadata_str,
    )
    if metadata_str:
        try:
            metadata = json.loads(metadata_str)
            if metadata.get("invite_bot") is True:
                logger.info("invite_bot=True → ACCEPTING job for room %s", req.room.name)
                await req.accept(
                    name="Beat",
                    identity="beat-facilitator",
                )
                return
        except json.JSONDecodeError:
            logger.warning("Failed to parse room metadata as JSON")

    logger.info("invite_bot not True → REJECTING job for room %s", req.room.name)
    await req.reject()


if __name__ == "__main__":
    agent_port = _resolve_agent_port()
    if _is_port_in_use(agent_port):
        logger.error(
            "Agent startup blocked: TCP port %s is already in use. "
            "Stop the existing agent process or set AGENT_PORT=0 for an ephemeral port.",
            agent_port,
        )
        sys.exit(1)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
            port=agent_port,
            agent_name="beat-facilitator",
        )
    )
