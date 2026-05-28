"""Emotion detection via Vokaturi SDK with acoustic-heuristic fallback."""

import logging
import os
import struct
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vokaturi SDK bootstrap
# ---------------------------------------------------------------------------

_vokaturi_api = settings.VOKATURI_PATH / "api"
sys.path.insert(0, str(_vokaturi_api))

try:
    import Vokaturi

    VOKATURI_AVAILABLE = True
    logger.info("Vokaturi module imported")
except ImportError:
    VOKATURI_AVAILABLE = False
    Vokaturi = None
    logger.warning("Vokaturi not available — fallback analysis will be used")


def _resolve_vokaturi_lib() -> Path:
    base = settings.VOKATURI_PATH / "lib" / "open"
    if sys.platform == "win32":
        suffix = "win32.dll" if struct.calcsize("P") == 4 else "win64.dll"
        return base / "win" / f"OpenVokaturi-4-0-{suffix}"
    if sys.platform == "darwin":
        return base / "macos" / "OpenVokaturi-4-0-mac.dylib"
    return base / "linux" / "OpenVokaturi-4-0-linux64.so"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class EmotionDetector:
    def __init__(self):
        self.vokaturi_loaded = False
        if not VOKATURI_AVAILABLE:
            return

        lib_path = _resolve_vokaturi_lib()
        if not lib_path.exists():
            logger.warning("Vokaturi lib not found at %s", lib_path)
            return
        try:
            Vokaturi.load(str(lib_path))
            self.vokaturi_loaded = True
            logger.info("Vokaturi loaded: %s", lib_path)
        except Exception as exc:
            logger.warning("Vokaturi load failed: %s", exc)

    def analyze_frame(self, frame: np.ndarray, sample_rate: int) -> Dict[str, float]:
        if not self.vokaturi_loaded:
            return _fallback(frame)

        try:
            buf = Vokaturi.float64array(len(frame))
            if frame.dtype == np.int16:
                buf[:] = frame[:] / 32768.0
            elif frame.dtype == np.int32:
                buf[:] = frame[:] / 2147483648.0
            else:
                buf[:] = frame[:]

            voice = Vokaturi.Voice(float(sample_rate), len(frame), 0)
            voice.fill_float64array(len(frame), buf)

            quality = Vokaturi.Quality()
            probs = Vokaturi.EmotionProbabilities()
            voice.extract(quality, probs)
            voice.destroy()

            if quality.valid:
                result = {
                    "neutral": probs.neutrality,
                    "happy": probs.happiness,
                    "sad": probs.sadness,
                    "angry": probs.anger,
                    "fearful": probs.fear,
                }
                logger.debug("[VOKATURI] %s", result)
                return result

            logger.debug("[VOKATURI] quality invalid, falling back")
        except Exception as exc:
            logger.warning("Vokaturi error: %s", exc)

        return _fallback(frame)


# ---------------------------------------------------------------------------
# Acoustic-heuristic fallback
# ---------------------------------------------------------------------------


def _fallback(frame: np.ndarray) -> Dict[str, float]:
    energy = float(np.sqrt(np.mean(frame**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))))

    if energy > 0.08:
        if zcr > 0.15:
            return {
                "neutral": 0.2,
                "happy": 0.5,
                "sad": 0.1,
                "angry": 0.1,
                "fearful": 0.1,
            }
        return {"neutral": 0.2, "happy": 0.1, "sad": 0.1, "angry": 0.4, "fearful": 0.2}

    if energy < 0.02:
        return {"neutral": 0.4, "happy": 0.1, "sad": 0.3, "angry": 0.1, "fearful": 0.1}

    return {"neutral": 0.6, "happy": 0.1, "sad": 0.1, "angry": 0.1, "fearful": 0.1}
