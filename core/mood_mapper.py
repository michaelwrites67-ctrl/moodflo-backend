"""Map Vokaturi emotions to Moodflo display categories."""

import logging
from typing import Dict

from config import settings

logger = logging.getLogger(__name__)


class MoodMapper:
    @staticmethod
    def map_emotion_to_category(emotion_dict: Dict[str, float], energy: float) -> str:
        happy = emotion_dict.get("happy", 0)
        angry = emotion_dict.get("angry", 0)
        fearful = emotion_dict.get("fearful", 0)
        sad = emotion_dict.get("sad", 0)
        neutral = emotion_dict.get("neutral", 0)
        threat = angry + fearful

        logger.debug(
            "[MAPPER] happy=%.3f angry=%.3f fear=%.3f sad=%.3f neutral=%.3f energy=%.1f threat=%.3f",
            happy,
            angry,
            fearful,
            sad,
            neutral,
            energy,
            threat,
        )

        # Energised/positive: genuine happiness signal at any energy level
        if happy >= 0.35:
            return "energised"

        # Stressed/tense: significant threat signal with any real audio
        if threat >= 0.50 and energy >= 2:
            return "stressed"

        # Flat/disengaged: very low energy with mostly neutral or sad tone
        if energy < 5 and (neutral >= 0.60 or sad >= 0.50):
            return "flat"

        # Flat: dominant sadness at any low-to-moderate energy
        if sad >= 0.55 and energy < 12:
            return "flat"

        # Thoughtful/constructive: calm, neutral engagement
        if neutral >= 0.40:
            return "thoughtful"

        # Stressed at moderate energy with elevated threat
        if threat >= 0.35 and energy >= 4:
            return "stressed"

        return "thoughtful"

    @staticmethod
    def get_category_display(category: str) -> str:
        return settings.MOODFLO_CATEGORIES.get(category, category)
