"""Moodflo extension backend — configuration."""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    CORS_ORIGINS: List[str] = ["*"]

    AUDIO_SAMPLE_RATE: int = 16000
    ENERGY_SCALE: int = 100
    EXTENSION_STREAM_BUFFER_FRAMES: int = 180

    SILENCE_ENERGY_THRESHOLD: float = 5.0
    SPEECH_ACTIVITY_ENERGY_THRESHOLD: float = 5.0

    VOKATURI_PATH: Path = (
        Path(__file__).resolve().parent.parent / "OpenVokaturi-4-0" / "OpenVokaturi-4-0"
    )

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.4-nano"
    OPENAI_TIMEOUT_SECONDS: float = 25.0

    MOODFLO_CATEGORIES: dict = {
        "energised": "Energised",
        "stressed": "Stressed/Tense",
        "flat": "Flat/Disengaged",
        "thoughtful": "Thoughtful/Constructive",
        "volatile": "Volatile/Unstable",
    }

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
