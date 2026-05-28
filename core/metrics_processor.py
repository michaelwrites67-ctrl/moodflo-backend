"""Metrics helpers for energy and category timelines."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class MetricsProcessor:
    def calculate_emotion_shifts(self, categories: List[str]) -> int:
        if len(categories) < 2:
            return 0
        return sum(
            1 for i in range(1, len(categories)) if categories[i] != categories[i - 1]
        )
