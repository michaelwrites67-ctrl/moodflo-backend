"""
Audio Processing Module
Handles audio extraction and segmentation for both batch and streaming
"""
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Tuple, Generator
from config import settings


class AudioProcessor:
    """Process audio from video/audio files"""
    
    def __init__(self, sample_rate: int = None):
        self.sample_rate = sample_rate or settings.AUDIO_SAMPLE_RATE
        self.frame_duration = settings.FRAME_DURATION
        self.hop_duration = settings.HOP_DURATION
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file using ffmpeg"""
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"extracted_{Path(video_path).stem}.wav")
        
        command = [
            'ffmpeg', '-i', str(video_path),
            '-ac', '1',  # Mono
            '-ar', str(self.sample_rate),
            '-vn',  # No video
            '-y',  # Overwrite
            output_path
        ]
        
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return output_path
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample if needed"""
        audio, sr = sf.read(file_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )
            except ImportError:
                raise ImportError("librosa required for resampling")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        return audio, self.sample_rate
    
    def segment_audio(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment audio into overlapping frames
        Returns: (frames, timestamps)
        """
        win_samples = int(self.frame_duration * self.sample_rate)
        hop_samples = int(self.hop_duration * self.sample_rate)
        
        frames = []
        timestamps = []
        
        for i in range(0, len(audio) - win_samples + 1, hop_samples):
            frame = audio[i:i + win_samples]
            frames.append(frame)
            timestamps.append(i / self.sample_rate)
        
        return np.array(frames), np.array(timestamps)
    
    def segment_audio_streaming(
        self,
        audio: np.ndarray,
        chunk_size: int = None
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Stream audio in chunks for real-time processing
        Yields: (frame, timestamp)
        """
        if chunk_size is None:
            chunk_size = int(settings.STREAM_UPDATE_INTERVAL * self.sample_rate)
        
        win_samples = int(self.frame_duration * self.sample_rate)
        
        for i in range(0, len(audio) - win_samples + 1, chunk_size):
            frame = audio[i:i + win_samples]
            timestamp = i / self.sample_rate
            yield frame, timestamp
    
    def compute_rms(self, frame: np.ndarray) -> float:
        """Compute RMS energy of frame"""
        return float(np.sqrt(np.mean(frame ** 2)))
    
    def is_silent(self, frame: np.ndarray) -> bool:
        """Check if frame is silent"""
        return self.compute_rms(frame) < settings.SILENCE_THRESHOLD
    
    def process_file(self, file_path: str) -> dict:
        """
        Process entire file for batch analysis
        Returns: audio data with frames, timestamps, duration
        """
        # Check file type
        file_ext = Path(file_path).suffix.lower()
        
        # Extract audio if video
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            audio_path = self.extract_audio_from_video(file_path)
            audio, sr = self.load_audio(audio_path)
            os.remove(audio_path)  # Clean up
        else:
            audio, sr = self.load_audio(file_path)
        
        # Segment into frames
        frames, timestamps = self.segment_audio(audio)
        
        return {
            'frames': frames,
            'timestamps': timestamps,
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'full_audio': audio
        }
