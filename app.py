"""
Moodflo extension backend — FastAPI + WebSocket for live Vokaturi analysis.
Run: uvicorn app:app --host 0.0.0.0 --port 8000 (cwd: backend/)
"""

import base64
import json
import logging
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure imports resolve when running as `uvicorn app:app` from backend/
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from config import settings
from services.extension_realtime_service import ExtensionRealtimeSession
from services.openai_structured_service import StructuredAiService
from services.pdf_report_service import PdfReportService

app = FastAPI(title="Moodflo Extension API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_service = StructuredAiService()
pdf_report_service = PdfReportService()


class FlowSuggestionRequest(BaseModel):
    team_tone: str = "Calibrating"
    speaking_energy: float = Field(default=0.0, ge=0.0, le=100.0)
    average_speaking_energy: float = Field(default=0.0, ge=0.0, le=100.0)
    share_of_voice: Dict[str, Any] = Field(default_factory=dict)
    tone_stability: float = Field(default=0.0, ge=0.0, le=100.0)
    silence_level: float = Field(default=0.0, ge=0.0, le=100.0)
    speaking_balance: str = "moderate"
    risk_level: str = "medium"


class FlowSuggestionResponse(BaseModel):
    suggestion: str


class MeetingSummaryRequest(BaseModel):
    meeting_title: str
    room_name: str = ""
    date: str
    start_time: str
    end_time: str
    total_duration_seconds: float = Field(default=0.0, ge=0.0)

    final_team_tone: str = "Calibrating"
    average_speaking_energy: float = Field(default=0.0, ge=0.0, le=100.0)
    tone_stability: float = Field(default=0.0, ge=0.0, le=100.0)
    share_of_voice: Dict[str, Any] = Field(default_factory=dict)
    speaking_balance: str = "moderate"
    silence_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_level: str = "medium"
    room_tone_profile: Dict[str, float] = Field(default_factory=dict)
    key_changes: Dict[str, Any] = Field(default_factory=dict)
    meeting_health_observations: List[str] = Field(default_factory=list)


class MeetingSummaryResponse(BaseModel):
    summary: str
    key_observations: List[str]
    suggested_next_steps: List[str]
    risk_level: str
    action_list: List[str] = Field(default_factory=list)


class EndMeetingPdfRequest(MeetingSummaryRequest):
    end_summary: Optional[str] = None
    key_observations: List[str] = Field(default_factory=list)
    recommended_next_steps: List[str] = Field(default_factory=list)
    action_list: List[str] = Field(default_factory=list)


class EndMeetingPdfResponse(BaseModel):
    filename: str
    pdf_base64: str
    summary: MeetingSummaryResponse


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "moodflo-extension-backend"}


@app.post("/api/ai/flow-suggestion", response_model=FlowSuggestionResponse)
async def flow_suggestion(payload: FlowSuggestionRequest):
    suggestion = ai_service.generate_flow_suggestion(payload.model_dump())
    return FlowSuggestionResponse(**suggestion)


@app.post("/api/ai/meeting-summary", response_model=MeetingSummaryResponse)
async def meeting_summary(payload: MeetingSummaryRequest):
    summary = ai_service.generate_meeting_summary(payload.model_dump())
    return MeetingSummaryResponse(**summary)


@app.post("/api/reports/end-meeting-pdf", response_model=EndMeetingPdfResponse)
async def end_meeting_pdf(payload: EndMeetingPdfRequest):
    payload_data = payload.model_dump()

    summary_data = None
    if (
        payload.end_summary
        and payload.key_observations
        and payload.recommended_next_steps
    ):
        summary_data = {
            "summary": payload.end_summary,
            "key_observations": payload.key_observations,
            "suggested_next_steps": payload.recommended_next_steps,
            "risk_level": payload.risk_level,
            "action_list": payload.action_list,
        }
    else:
        summary_data = ai_service.generate_meeting_summary(payload_data)

    payload_data["end_summary"] = summary_data["summary"]
    payload_data["key_observations"] = summary_data["key_observations"]
    payload_data["recommended_next_steps"] = summary_data["suggested_next_steps"]
    payload_data["risk_level"] = summary_data["risk_level"]
    payload_data["action_list"] = summary_data.get("action_list", [])

    total_seconds = float(payload_data.get("total_duration_seconds") or 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    payload_data["total_duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    pdf_bytes = pdf_report_service.build_end_of_meeting_pdf(payload_data)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    filename = pdf_report_service.build_filename(payload.meeting_title, payload.date)

    return EndMeetingPdfResponse(
        filename=filename,
        pdf_base64=pdf_b64,
        summary=MeetingSummaryResponse(**summary_data),
    )


@app.websocket("/ws/extension/live/{session_id}")
async def extension_live(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session_state: Optional[ExtensionRealtimeSession] = None
    handshake_ok = False
    sample_rate = settings.AUDIO_SAMPLE_RATE

    logger.info("[WS] Connection accepted: session=%s", session_id)

    try:
        while True:
            raw = await websocket.receive()
            if raw["type"] == "websocket.disconnect":
                logger.info("[WS] Client disconnected: session=%s", session_id)
                break

            if raw["type"] != "websocket.receive":
                continue

            if "text" in raw and raw["text"]:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    logger.warning("[WS] Invalid JSON received")
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON"}
                    )
                    continue

                mtype = msg.get("type")
                logger.debug("[WS] Text message: type=%s", mtype)

                if mtype == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if mtype == "start":
                    sample_rate = int(
                        msg.get("sample_rate", settings.AUDIO_SAMPLE_RATE)
                    )
                    mode = msg.get("mode", "single")
                    session_state = ExtensionRealtimeSession(session_id)
                    handshake_ok = True
                    logger.info(
                        "[WS] Handshake OK: session=%s sample_rate=%d mode=%s",
                        session_id,
                        sample_rate,
                        mode,
                    )
                    await websocket.send_json(
                        {
                            "type": "ready",
                            "session_id": session_id,
                            "sample_rate": sample_rate,
                            "mode": mode,
                            "message": "Handshake OK — send speaker-labeled PCM frames",
                        }
                    )
                    continue

                if mtype == "speaker_removed":
                    speaker_id = msg.get("speaker_id", "")
                    if session_state and speaker_id:
                        session_state.remove_speaker(speaker_id)
                        logger.info("[WS] Speaker removed: %s", speaker_id)
                    continue

                await websocket.send_json(
                    {"type": "error", "message": f"Unknown control message: {mtype}"}
                )
                continue

            if "bytes" in raw and raw["bytes"]:
                if not handshake_ok or session_state is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Send {type:'start', sample_rate} first",
                        }
                    )
                    continue

                data = raw["bytes"]

                # Parse multi-speaker binary protocol:
                # [4 bytes: header_len LE][header JSON][PCM int16 data]
                speaker_id = "unknown"
                is_local = False
                pcm_start = 0

                if len(data) > 4:
                    try:
                        header_len = struct.unpack("<I", data[:4])[0]
                        if 0 < header_len < len(data) - 4:
                            header_json = data[4 : 4 + header_len].decode("utf-8")
                            header = json.loads(header_json)
                            speaker_id = header.get("s", "unknown")
                            is_local = bool(header.get("l", 0))
                            pcm_start = 4 + header_len
                        else:
                            # Fallback: treat entire data as PCM (legacy format)
                            pcm_start = 0
                            speaker_id = "mixed_legacy"
                    except (struct.error, json.JSONDecodeError, UnicodeDecodeError):
                        # Fallback: legacy raw PCM without header
                        pcm_start = 0
                        speaker_id = "mixed_legacy"
                        logger.debug("[WS] Binary without header, using legacy mode")

                pcm = np.frombuffer(data[pcm_start:], dtype=np.int16)

                if len(pcm) > 0:
                    rms_val = np.sqrt(np.mean(pcm.astype(np.float64) ** 2))
                    logger.debug(
                        "[WS] PCM chunk: speaker=%s local=%s size=%d rms=%.2f max=%d min=%d",
                        speaker_id,
                        is_local,
                        len(pcm),
                        rms_val,
                        np.max(pcm),
                        np.min(pcm),
                    )
                else:
                    logger.debug("[WS] Empty PCM chunk from speaker=%s", speaker_id)

                result = session_state.process_speaker_pcm(
                    speaker_id, is_local, pcm, sample_rate
                )
                if not result:
                    continue

                await websocket.send_json(
                    {
                        "type": "emotion_update",
                        "time": result["time"],
                        "data": result["data"],
                    }
                )

    except WebSocketDisconnect:
        logger.info("[WS] WebSocket disconnected: session=%s", session_id)
    except Exception as e:
        logger.exception("[WS] Unexpected error in session=%s: %s", session_id, e)
    finally:
        if session_state:
            logger.info(
                "[WS] Session ended: %s — speakers tracked: %d",
                session_id,
                len(session_state.speakers),
            )
        session_state = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
