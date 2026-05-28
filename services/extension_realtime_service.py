"""
Multi-speaker realtime metrics for Chrome extension.

Each speaker's audio is processed individually through Vokaturi.
Results are emitted immediately on every PCM chunk (no timing gates, no smoothing).
"""

import logging
import time
from collections import deque
from typing import Dict, List

import numpy as np

from config import settings
from core.emotion_detector import EmotionDetector
from core.metrics_processor import MetricsProcessor
from core.mood_mapper import MoodMapper

logger = logging.getLogger(__name__)


class SpeakerState:
    """Per-speaker tracking state."""

    def __init__(self, speaker_id: str, is_local: bool):
        self.speaker_id = speaker_id
        self.is_local = is_local
        self.energy_values: deque = deque(
            maxlen=settings.EXTENSION_STREAM_BUFFER_FRAMES
        )
        self.categories: deque = deque(maxlen=settings.EXTENSION_STREAM_BUFFER_FRAMES)
        self.last_emotion: Dict[str, float] = {}
        self.last_energy: float = 0.0
        self.last_category: str = "Thoughtful/Constructive"
        self.frame_count: int = 0
        self.first_seen: float = time.monotonic()
        self.has_audio: bool = False  # True once first meaningful audio frame received
        logger.info("[SPEAKER] Created state for %s (local=%s)", speaker_id, is_local)


class ExtensionRealtimeSession:
    """One session per WebSocket connection — multi-speaker, real-time emission."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.detector = EmotionDetector()
        self.mapper = MoodMapper()
        self.metrics = MetricsProcessor()
        self.speakers: Dict[str, SpeakerState] = {}
        self._t0 = time.monotonic()
        logger.info("[SESSION] Created session %s", session_id)

    def process_speaker_pcm(
        self,
        speaker_id: str,
        is_local: bool,
        pcm_int16: np.ndarray,
        sample_rate: int,
    ) -> Dict:
        """Process a PCM chunk for a single speaker. Returns result immediately."""
        if pcm_int16.size == 0:
            logger.debug("[SESSION] Empty PCM from speaker %s, skipping", speaker_id)
            return {}

        if pcm_int16.dtype != np.int16:
            pcm_int16 = pcm_int16.astype(np.int16)

        # Get or create speaker state
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = SpeakerState(speaker_id, is_local)
        speaker = self.speakers[speaker_id]
        speaker.frame_count += 1

        # Compute energy
        pcm_f = pcm_int16.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(pcm_f**2)))
        energy = min(rms * settings.ENERGY_SCALE, 100.0)

        # Run Vokaturi on this speaker's audio
        emotions = self.detector.analyze_frame(pcm_int16, sample_rate)
        cat_key = self.mapper.map_emotion_to_category(emotions, energy)
        display = self.mapper.get_category_display(cat_key)

        # Update speaker state
        speaker.energy_values.append(energy)
        speaker.categories.append(display)
        speaker.last_emotion = emotions
        speaker.last_energy = energy
        speaker.last_category = display
        if energy >= settings.SPEECH_ACTIVITY_ENERGY_THRESHOLD:
            speaker.has_audio = True

        logger.debug(
            "[SESSION] Speaker %s frame=%d energy=%.1f emotion=%s raw=%s",
            speaker_id,
            speaker.frame_count,
            energy,
            display,
            emotions,
        )

        # Build per-speaker result
        t = time.monotonic() - self._t0
        speaker_result = {
            "speaker_id": speaker_id,
            "is_local": is_local,
            "emotion": display,
            "emotions_raw": emotions,
            "energy": round(energy, 2),
            "frame_count": speaker.frame_count,
        }

        # Build aggregated room-level metrics
        room = self._build_room_metrics(t)

        result = {
            "time": t,
            "data": {
                "time": t,
                "speaker": speaker_result,
                # Only count speakers that have ever produced meaningful audio
                "speaker_count": sum(1 for s in self.speakers.values() if s.has_audio),
                "active_speaker_count": sum(
                    1
                    for s in self.speakers.values()
                    if s.has_audio
                    and s.last_energy >= settings.SPEECH_ACTIVITY_ENERGY_THRESHOLD
                ),
                "speakers": {
                    sid: {
                        "emotion": s.last_category,
                        "energy": round(s.last_energy, 2),
                        "is_local": s.is_local,
                    }
                    for sid, s in self.speakers.items()
                    if s.has_audio
                },
                **room,
            },
        }

        logger.debug(
            "[SESSION] Emitting result: speakers=%d dominant=%s energy=%.1f participation=%.1f",
            len(self.speakers),
            room.get("current_emotion"),
            room.get("current_energy", 0),
            room.get("participation", 0),
        )
        return result

    def remove_speaker(self, speaker_id: str) -> None:
        """Remove a speaker who left the call."""
        if speaker_id in self.speakers:
            del self.speakers[speaker_id]
            logger.info(
                "[SESSION] Removed speaker %s, remaining=%d",
                speaker_id,
                len(self.speakers),
            )

    def _build_room_metrics(self, t: float) -> Dict:
        """Aggregate metrics across all active speakers — raw values, no smoothing."""
        if not self.speakers:
            return {
                "current_emotion": "Calibrating",
                "current_energy": 0,
                "avg_energy": 0,
                "silence_percentage": 100,
                "emotion_shifts": 0,
                "volatility": 0,
                "participation": 0,
                "emotion_distribution": {},
                "sentiment_breakdown": {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "silence": 100,
                    "input": 0,
                },
                "timeline_length": 0,
                "is_processed": True,
            }

        # Collect recent data from all speakers (last 10 frames each)
        all_energies: List[float] = []
        all_categories: List[str] = []
        current_energies: List[float] = []

        # Only include speakers that have ever produced meaningful audio
        confirmed_speakers = [s for s in self.speakers.values() if s.has_audio]
        if not confirmed_speakers:
            # No confirmed speakers yet — fall back to all (early session)
            confirmed_speakers = list(self.speakers.values())

        # Among confirmed speakers, prioritise those actively speaking now
        active_speakers = [
            s
            for s in confirmed_speakers
            if s.last_energy >= settings.SPEECH_ACTIVITY_ENERGY_THRESHOLD
        ]
        contributing = active_speakers if active_speakers else confirmed_speakers

        for speaker in contributing:
            recent_e = list(speaker.energy_values)[-10:]
            recent_c = list(speaker.categories)[-10:]
            all_energies.extend(recent_e)
            all_categories.extend(recent_c)
            current_energies.append(speaker.last_energy)

        # Energy
        current_energy = float(np.mean(current_energies)) if current_energies else 0.0
        avg_energy = float(np.mean(all_energies)) if all_energies else 0.0

        # Participation: % of confirmed speakers actively speaking
        active = sum(
            1
            for s in confirmed_speakers
            if s.last_energy >= settings.SPEECH_ACTIVITY_ENERGY_THRESHOLD
        )
        total = len(confirmed_speakers)
        participation = (active / total * 100) if total > 0 else 0.0

        # Silence
        silent_count = sum(
            1 for e in all_energies if e < settings.SILENCE_ENERGY_THRESHOLD
        )
        silence_pct = (
            (silent_count / len(all_energies) * 100) if all_energies else 100.0
        )

        # Category distribution
        dist: Dict[str, float] = {}
        for cat in all_categories:
            dist[cat] = dist.get(cat, 0) + 1
        total_cats = len(all_categories)
        distribution = (
            {k: (v / total_cats) * 100 for k, v in dist.items()}
            if total_cats > 0
            else {}
        )

        # Dominant emotion
        dominant = (
            max(distribution.items(), key=lambda x: x[1])[0]
            if distribution
            else "Calibrating"
        )

        # Volatility
        volatility = 0.0
        if len(all_energies) > 1:
            volatility = min(float(np.std(np.diff(all_energies))) / 5.0, 10.0)

        # Emotion shifts
        emotion_shifts = self.metrics.calculate_emotion_shifts(all_categories)

        # Sentiment breakdown
        sentiment = self._build_sentiment_breakdown(distribution, silence_pct)

        return {
            "current_emotion": dominant,
            "current_energy": round(current_energy, 2),
            "avg_energy": round(avg_energy, 2),
            "silence_percentage": round(silence_pct, 2),
            "emotion_shifts": emotion_shifts,
            "volatility": round(volatility, 2),
            "participation": round(participation, 2),
            "emotion_distribution": distribution,
            "sentiment_breakdown": sentiment,
            "timeline_length": sum(
                len(s.energy_values) for s in self.speakers.values()
            ),
            "is_processed": True,
        }

    @staticmethod
    def _build_sentiment_breakdown(
        distribution: Dict[str, float],
        silence_pct: float,
    ) -> Dict[str, float]:
        silence_pct = max(0.0, min(100.0, float(silence_pct)))
        input_pct = max(0.0, 100.0 - silence_pct)

        positive_raw = distribution.get("Energised", 0.0)
        neutral_raw = distribution.get("Thoughtful/Constructive", 0.0)
        negative_raw = (
            distribution.get("Stressed/Tense", 0.0)
            + distribution.get("Volatile/Unstable", 0.0)
            + distribution.get("Flat/Disengaged", 0.0)
        )

        active_total = max(0.001, positive_raw + neutral_raw + negative_raw)
        positive = input_pct * (positive_raw / active_total)
        neutral = input_pct * (neutral_raw / active_total)
        negative = input_pct * (negative_raw / active_total)

        return {
            "positive": round(max(0.0, min(100.0, positive)), 2),
            "neutral": round(max(0.0, min(100.0, neutral)), 2),
            "negative": round(max(0.0, min(100.0, negative)), 2),
            "silence": round(silence_pct, 2),
            "input": round(input_pct, 2),
        }
