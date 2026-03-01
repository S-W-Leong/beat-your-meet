import logging
import os
import json
import secrets
import string
import re as _re
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from livekit import api
from mistralai import Mistral
from dotenv import load_dotenv

# Resolve absolute path to .env so it works regardless of CWD or how Python
# was invoked (e.g. `python main.py` vs `uvicorn main:app` from project root).
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_dotenv_path)

logger = logging.getLogger("beat-your-meet-server")
logger.setLevel(logging.INFO)

_DATA_DIR = Path(__file__).resolve().parent / "data"

app = FastAPI(title="Beat Your Meet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))


# ── Models ────────────────────────────────────────────────────────────


class TokenRequest(BaseModel):
    room_name: str
    participant_name: str
    access_code: str


class AgendaRequest(BaseModel):
    description: str
    duration_minutes: int


class CreateRoomRequest(BaseModel):
    agenda: dict
    style: str  # "gentle" | "moderate" | "chatting"
    invite_bot: bool = True


class BotControlRequest(BaseModel):
    host_token: str


class UploadDocRequest(BaseModel):
    filename: str
    title: str
    content: str


class DocMeta(BaseModel):
    filename: str
    title: str
    size_bytes: int
    created_at: str


_ROOM_ID_RE = _re.compile(r"^meet-[a-f0-9]{8}$")
_FILENAME_RE = _re.compile(r"^[a-z0-9][a-z0-9-]{0,58}\.md$")


def _safe_room_dir(room_id: str) -> Path:
    """Validate room_id and return its data directory. Raises 400 on invalid input."""
    if not _ROOM_ID_RE.match(room_id):
        raise HTTPException(status_code=400, detail="Invalid room_id format")
    return _DATA_DIR / "rooms" / room_id


def _safe_doc_path(room_id: str, filename: str) -> Path:
    """Validate both room_id and filename, return the full path."""
    if not _FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename format")
    return _safe_room_dir(room_id) / filename


# ── Token Generation ─────────────────────────────────────────────────


@app.post("/api/token")
async def generate_token(req: TokenRequest):
    try:
        # Validate access code against room metadata
        lk_api = api.LiveKitAPI(
            os.environ["LIVEKIT_URL"],
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
        try:
            rooms = await lk_api.room.list_rooms(api.ListRoomsRequest(names=[req.room_name]))
            if not rooms.rooms:
                raise HTTPException(status_code=404, detail="Room not found or has ended")
            room_metadata = json.loads(rooms.rooms[0].metadata or "{}")
            stored_code = room_metadata.get("access_code", "")
            if stored_code.upper() != req.access_code.upper():
                logger.warning(f"Invalid access code attempt for room: {req.room_name}")
                raise HTTPException(status_code=403, detail="Invalid access code")
        finally:
            await lk_api.aclose()

        token = api.AccessToken(
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
        token.with_identity(req.participant_name)
        token.with_name(req.participant_name)
        token.with_grants(
            api.VideoGrants(
                room_join=True,
                room=req.room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        return {"token": token.to_jwt()}
    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Missing env var for token generation: {e}")
        raise HTTPException(status_code=500, detail=f"Server misconfigured: missing {e}")
    except Exception as e:
        logger.exception("Token generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Agenda Generation ────────────────────────────────────────────────


AGENDA_GENERATION_PROMPT = """Based on the meeting description below, generate a structured agenda.

Meeting Description:
{description}

Total meeting duration: {duration_minutes} minutes

Generate a JSON object with this exact structure:
{{
  "title": "Meeting title",
  "items": [
    {{
      "id": 1,
      "topic": "Topic name",
      "description": "Brief description of what to cover",
      "duration_minutes": 10
    }}
  ],
  "total_minutes": {duration_minutes}
}}

Rules:
- Keep the total duration of all items within {duration_minutes} minutes
- Each item should have a clear, concise topic name
- Order items by priority (most important first)
- Be realistic about time — discussion takes longer than you think
- Aim for 3-6 items depending on the total duration
"""


@app.post("/api/agenda")
async def generate_agenda(req: AgendaRequest):
    if not os.environ.get("MISTRAL_API_KEY"):
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not configured")

    try:
        response = await mistral_client.chat.complete_async(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "system",
                    "content": "You are a meeting planning assistant. Generate structured agendas in JSON format. Only output valid JSON.",
                },
                {
                    "role": "user",
                    "content": AGENDA_GENERATION_PROMPT.format(
                        description=req.description,
                        duration_minutes=req.duration_minutes,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1024,
        )

        agenda = json.loads(response.choices[0].message.content)
        return agenda
    except json.JSONDecodeError as e:
        logger.error(f"Mistral returned invalid JSON for agenda: {e}")
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON")
    except Exception as e:
        logger.exception("Agenda generation failed")
        raise HTTPException(status_code=502, detail=f"Agenda generation failed: {e}")


# ── Room Creation ────────────────────────────────────────────────────


def generate_access_code() -> str:
    chars = string.ascii_uppercase + string.digits
    code = "".join(secrets.choice(chars) for _ in range(4))
    return f"MEET-{code}"


@app.post("/api/room")
async def create_room(req: CreateRoomRequest):
    import uuid

    logger.info(f"create_room called — invite_bot={req.invite_bot}, style={req.style}")
    room_name = f"meet-{uuid.uuid4().hex[:8]}"
    access_code = generate_access_code()
    host_token = secrets.token_hex(16)

    # Store room metadata (agenda + style + access code + host token) in LiveKit room metadata.
    # Include invite_bot so the agent can check whether it should accept the job.
    room_metadata = json.dumps(
        {
            "agenda": req.agenda,
            "style": req.style,
            "access_code": access_code,
            "host_token": host_token,
            "invite_bot": req.invite_bot,
        }
    )

    try:
        lk_api = api.LiveKitAPI(
            os.environ["LIVEKIT_URL"],
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
        try:
            await lk_api.room.create_room(
                api.CreateRoomRequest(
                    name=room_name,
                    metadata=room_metadata,
                )
            )

            # Only explicitly dispatch the agent when the host opted in
            if req.invite_bot:
                await lk_api.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(
                        agent_name="beat-facilitator",
                        room=room_name,
                    )
                )
                logger.info(f"Dispatched bot to room: {room_name}")
        finally:
            await lk_api.aclose()
    except KeyError as e:
        logger.error(f"Missing env var for room creation: {e}")
        raise HTTPException(status_code=500, detail=f"Server misconfigured: missing {e}")
    except Exception as e:
        logger.exception("Room creation failed")
        raise HTTPException(status_code=502, detail=f"Failed to create LiveKit room: {e}")

    logger.info(f"Created room: {room_name} with access code: {access_code}")
    return {"room_name": room_name, "access_code": access_code, "host_token": host_token}


# ── Bot Control ──────────────────────────────────────────────────────


async def _verify_host_token(room_name: str, host_token: str) -> None:
    """Verify that the provided host_token matches the one stored in room metadata."""
    lk_api = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        rooms = await lk_api.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
        if not rooms.rooms:
            raise HTTPException(status_code=404, detail="Room not found")
        room_metadata = json.loads(rooms.rooms[0].metadata or "{}")
        stored_token = room_metadata.get("host_token", "")
        if not secrets.compare_digest(stored_token, host_token):
            raise HTTPException(status_code=403, detail="Invalid host token")
    finally:
        await lk_api.aclose()


@app.post("/api/room/{room_name}/invite-bot")
async def invite_bot(room_name: str, req: BotControlRequest):
    logger.info(f"invite-bot called for room={room_name}, host_token={'present' if req.host_token else 'missing'}")
    await _verify_host_token(room_name, req.host_token)

    lk_api = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        # Check if bot is already in the room
        participants = await lk_api.room.list_participants(
            api.ListParticipantsRequest(room=room_name)
        )
        for p in participants.participants:
            if p.identity == "beat-facilitator":
                return {"status": "already_active"}

        # Update room metadata to set invite_bot=True so the agent's
        # request_fnc will accept the job.
        rooms = await lk_api.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
        if rooms.rooms:
            metadata = json.loads(rooms.rooms[0].metadata or "{}")
            metadata["invite_bot"] = True
            await lk_api.room.update_room_metadata(
                api.UpdateRoomMetadataRequest(room=room_name, metadata=json.dumps(metadata))
            )

        # Dispatch the agent
        await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="beat-facilitator",
                room=room_name,
            )
        )
    finally:
        await lk_api.aclose()

    logger.info(f"Invited bot to room: {room_name}")
    return {"status": "invited"}


@app.delete("/api/room/{room_name}/bot")
async def remove_bot(room_name: str, req: BotControlRequest):
    await _verify_host_token(room_name, req.host_token)

    lk_api = api.LiveKitAPI(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        await lk_api.room.remove_participant(
            api.RoomParticipantIdentity(room=room_name, identity="beat-facilitator")
        )
    except Exception as e:
        logger.warning(f"Failed to remove bot from {room_name}: {e}")
        raise HTTPException(status_code=404, detail="Bot not found in room")
    finally:
        await lk_api.aclose()

    logger.info(f"Removed bot from room: {room_name}")
    return {"status": "removed"}


# ── Document Storage ──────────────────────────────────────────────────


@app.post("/api/rooms/{room_id}/docs", status_code=201)
async def upload_doc(room_id: str, req: UploadDocRequest):
    """Agent uploads a generated markdown document."""
    path = _safe_doc_path(room_id, req.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(req.content, encoding="utf-8")
    meta_path = path.with_suffix(".meta.json")
    meta = {
        "filename": req.filename,
        "title": req.title,
        "size_bytes": len(req.content.encode("utf-8")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    logger.info("Stored doc %s for room %s", req.filename, room_id)
    return {"ok": True}


@app.get("/api/rooms/{room_id}/docs")
async def list_docs(room_id: str) -> list[DocMeta]:
    """List all available documents for a room."""
    room_dir = _safe_room_dir(room_id)
    if not room_dir.exists():
        return []
    docs = []
    for meta_path in sorted(room_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            docs.append(DocMeta(**meta))
        except Exception:
            logger.warning("Could not read meta file %s", meta_path)
    return docs


@app.get("/api/rooms/{room_id}/docs/{filename}")
async def get_doc(room_id: str, filename: str) -> dict:
    """Fetch the raw markdown content of a specific document."""
    path = _safe_doc_path(room_id, filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    content = path.read_text(encoding="utf-8")
    meta_path = path.with_suffix(".meta.json")
    title = filename
    if meta_path.exists():
        try:
            title = json.loads(meta_path.read_text(encoding="utf-8"))["title"]
        except Exception:
            pass
    return {"filename": filename, "title": title, "content": content}


# ── Health Check ─────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
