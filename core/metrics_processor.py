"""
Metrics Processing Module
Calculate meeting metrics from audio and emotion data
"""
import numpy as np
from typing import Dict, List
from config import settings


class MetricsProcessor:
    """Calculate meeting performance metrics"""
    
    def __init__(self, sample_rate: int = None):
        self.sample_rate = sample_rate or settings.AUDIO_SAMPLE_RATE
    
    def calculate_energy_timeline(self, frames: np.ndarray) -> List[float]:
        """Calculate energy level for each frame"""
        energy_timeline = []
        
        for frame in frames:
            rms = np.sqrt(np.mean(frame ** 2))
            # Scale to 0-100
            energy = min(rms * settings.ENERGY_SCALE, 100)
            energy_timeline.append(float(energy))
        
        return energy_timeline
    
    def calculate_silence_percentage(
        self,
        frames: np.ndarray,
        threshold: float = None
    ) -> float:
        """Calculate percentage of silent frames"""
        if threshold is None:
            threshold = settings.SILENCE_THRESHOLD
        
        silent_count = 0
        for frame in frames:
            # Calculate RMS energy
            rms = np.sqrt(np.mean(frame ** 2))
            # Convert to more sensitive scale (0-1 range for audio)
            energy_level = rms
            
            # Consider silence if energy is below threshold
            # Typical speech has energy > 0.01, silence < 0.005
            if energy_level < threshold:
                silent_count += 1
        
        silence_pct = (silent_count / len(frames)) * 100
        return float(silence_pct)
    
    def calculate_participation(
        self,
        frames: np.ndarray,
        threshold: float = 0.02
    ) -> float:
        """
        Calculate participation percentage
        Based on RMS energy of frames (speech vs silence)
        """
        if len(frames) == 0:
            return 0.0
        
        active_count = 0
        for frame in frames:
            rms = np.sqrt(np.mean(frame ** 2))
            # Active speech typically has RMS > 0.02
            if rms > threshold:
                active_count += 1
        
        participation = (active_count / len(frames)) * 100
        return float(participation)
    
    def calculate_volatility(self, energy_timeline: List[float]) -> float:
        """
        Calculate emotional volatility
        Based on energy variations
        """
        if len(energy_timeline) < 2:
            return 0.0
        
        # Calculate differences
        diffs = np.diff(energy_timeline)
        
        # Standard deviation of changes
        volatility = np.std(diffs)
        
        # Scale to 0-10
        return float(min(volatility / 5, 10))
    
    def calculate_average_energy(self, energy_timeline: List[float]) -> float:
        """Calculate average energy level"""
        return float(np.mean(energy_timeline))
    
    def calculate_emotion_shifts(self, categories: List[str]) -> int:
        """Count number of emotion category changes"""
        if len(categories) < 2:
            return 0
        
        shifts = 0
        for i in range(1, len(categories)):
            if categories[i] != categories[i-1]:
                shifts += 1
        
        return shifts
    
    def calculate_all_metrics(
        self,
        frames: np.ndarray,
        emotion_series: List[Dict],
        full_audio: np.ndarray = None
    ) -> Dict:
        """
        Calculate all metrics at once
        Returns comprehensive metrics dictionary
        """
        # Energy timeline
        energy_timeline = self.calculate_energy_timeline(frames)
        
        # Silence percentage
        silence_pct = self.calculate_silence_percentage(frames)
        
        # Participation
        participation = self.calculate_participation(frames)
        
        # Average energy
        avg_energy = self.calculate_average_energy(energy_timeline)
        
        # Volatility
        volatility = self.calculate_volatility(energy_timeline)
        
        return {
            'energy_timeline': energy_timeline,
            'silence_percentage': silence_pct,
            'participation': participation,
            'avg_energy': avg_energy,
            'volatility': volatility
        }
    
    def calculate_realtime_metrics(
        self,
        energy_values: List[float],
        categories: List[str],
        current_time: float
    ) -> Dict:
        """
        Calculate metrics for real-time display
        Uses accumulated data up to current time
        """
        if len(energy_values) == 0:
            return {
                'avg_energy': 0,
                'silence_pct': 0,
                'emotion_shifts': 0,
                'volatility': 0
            }
        
        # Average energy
        avg_energy = float(np.mean(energy_values))
        
        # Silence (frames below 20)
        silent_count = sum(1 for e in energy_values if e < 20)
        silence_pct = (silent_count / len(energy_values)) * 100
        
        # Emotion shifts
        emotion_shifts = self.calculate_emotion_shifts(categories)
        
        # Volatility
        volatility = self.calculate_volatility(energy_values)
        
        return {
            'avg_energy': avg_energy,
            'silence_pct': silence_pct,
            'emotion_shifts': emotion_shifts,
            'volatility': volatility
        }
