"""OpenAI-backed structured outputs for coaching and meeting summaries."""

import json
import logging
from typing import Any, Dict, List

from config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

logger = logging.getLogger(__name__)


class StructuredAiService:
    """Generates stable JSON outputs for the extension's AI features."""

    _FLOW_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "suggestion": {"type": "string"},
        },
        "required": ["suggestion"],
    }

    _SUMMARY_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "key_observations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "suggested_next_steps": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risk_level": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
            "action_list": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "summary",
            "key_observations",
            "suggested_next_steps",
            "risk_level",
            "action_list",
        ],
    }

    def __init__(self):
        self._client = None

        if not settings.OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY not configured; AI endpoints use fallback logic."
            )
            return

        if OpenAI is None:
            logger.warning(
                "openai package is missing; AI endpoints use fallback logic."
            )
            return

        try:
            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.OPENAI_TIMEOUT_SECONDS,
            )
        except TypeError:
            # Older SDK versions may not support the timeout init arg.
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_risk_level(raw: Any) -> str:
        text = str(raw or "").strip().lower()
        if text in {"low", "medium", "high"}:
            return text
        return "medium"

    def _request_structured_json(
        self,
        schema_name: str,
        schema: Dict[str, Any],
        system_prompt: str,
        payload: Dict[str, Any],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError("OpenAI client is not configured")

        completion = self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=True),
                },
            ],
        )

        message = completion.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"OpenAI refusal: {refusal}")

        content = message.content
        if not content:
            raise RuntimeError("OpenAI returned empty content")

        if isinstance(content, list):
            parts = []
            for chunk in content:
                if isinstance(chunk, dict):
                    parts.append(str(chunk.get("text", "")))
                else:
                    parts.append(str(getattr(chunk, "text", "")))
            content = "".join(parts)

        return json.loads(content)

    def _fallback_flow(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        team_tone = str(metrics.get("team_tone") or "Calibrating")
        silence = self._to_float(metrics.get("silence_level"))
        speaking_balance = str(metrics.get("speaking_balance") or "moderate")

        share = metrics.get("share_of_voice") or {}
        participation = self._to_float(share.get("participation_pct"), default=50.0)

        if silence >= 45:
            return {"suggestion": "Invite a quick round of updates to cut the silence."}

        if participation <= 45:
            return {"suggestion": "Call on a quieter participant for their take."}

        if team_tone in {"Stressed/Tense", "Volatile/Unstable"}:
            return {"suggestion": "Pause and ask one clarifying question before moving on."}

        if speaking_balance.lower().startswith("balanced"):
            return {"suggestion": "Lock in momentum by confirming the next owner and deadline."}

        return {"suggestion": "Keep turns concise and rotate the floor once."}

    def generate_flow_suggestion(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are Moodflo Live Coach. You ONLY use acoustic meeting metrics. "
            "Do not mention transcript content, sentiment analysis, diagnosis, therapy, "
            "mental health, or personal traits. "
            "Return exactly ONE sentence of 20 words or fewer as a practical coaching action. "
            "No explanations, no follow-up — just the single actionable sentence. "
            "Output MUST match the provided JSON schema."
        )

        try:
            output = self._request_structured_json(
                schema_name="flow_suggestion",
                schema=self._FLOW_SCHEMA,
                system_prompt=system_prompt,
                payload=metrics,
                temperature=0.2,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Flow suggestion fallback used: %s", exc)
            output = self._fallback_flow(metrics)

        suggestion = str(output.get("suggestion") or "Keep turn-taking balanced.").strip()

        # Hard safety net: truncate to 20 words
        words = suggestion.split()
        if len(words) > 20:
            suggestion = " ".join(words[:20])
            if not suggestion.endswith("."):
                suggestion += "."

        return {"suggestion": suggestion}

    def _fallback_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        meeting_title = str(report.get("meeting_title") or "Meeting")
        duration_seconds = self._to_float(report.get("total_duration_seconds"), 0.0)
        final_tone = str(report.get("final_team_tone") or "Calibrating")
        avg_energy = self._to_float(report.get("average_speaking_energy"), 0.0)
        silence_pct = self._to_float(report.get("silence_percentage"), 0.0)
        risk_level = self._normalize_risk_level(report.get("risk_level"))

        minutes = int(round(duration_seconds / 60.0))
        summary = (
            f"{meeting_title} ran for about {minutes} minutes. "
            f"The final team tone was {final_tone}, with average speaking energy "
            f"at {avg_energy:.1f}/100 and silence at {silence_pct:.1f}%."
        )

        observations = [
            f"Final team tone settled at {final_tone}.",
            f"Silence remained around {silence_pct:.1f}% across the captured session.",
            f"Average speaking energy was {avg_energy:.1f}/100.",
        ]

        next_steps = [
            "Start the next meeting with a quick objective reset in the first two minutes.",
            "Use one timed round-robin if participation starts to narrow.",
            "Confirm owners and deadlines before closing.",
        ]

        action_list = [
            "Document one decision and one open question.",
            "Assign a facilitator for speaking balance in the next session.",
        ]

        return {
            "summary": summary,
            "key_observations": observations,
            "suggested_next_steps": next_steps,
            "risk_level": risk_level,
            "action_list": action_list,
        }

    def generate_meeting_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are Moodflo Meeting Summary. Use ONLY aggregate acoustic metrics and "
            "meeting dynamics. Do not mention transcript content, sentiment labels, mental "
            "health terms, diagnosis, or individual profiling. Keep output one para 3-5 lines concise and "
            "actionable. Output MUST match the JSON schema exactly."
        )

        try:
            output = self._request_structured_json(
                schema_name="meeting_summary",
                schema=self._SUMMARY_SCHEMA,
                system_prompt=system_prompt,
                payload=report,
                temperature=0.2,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Meeting summary fallback used: %s", exc)
            output = self._fallback_summary(report)

        result = {
            "summary": str(output.get("summary") or "").strip(),
            "key_observations": [
                str(x).strip() for x in output.get("key_observations", [])
            ],
            "suggested_next_steps": [
                str(x).strip() for x in output.get("suggested_next_steps", [])
            ],
            "risk_level": self._normalize_risk_level(output.get("risk_level")),
            "action_list": [str(x).strip() for x in output.get("action_list", [])],
        }

        # Ensure minimum useful shape even if model outputs sparse content.
        if not result["summary"]:
            fallback = self._fallback_summary(report)
            result["summary"] = fallback["summary"]

        if not result["key_observations"]:
            result["key_observations"] = self._fallback_summary(report)[
                "key_observations"
            ]

        if not result["suggested_next_steps"]:
            result["suggested_next_steps"] = self._fallback_summary(report)[
                "suggested_next_steps"
            ]

        return result
