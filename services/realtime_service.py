"""
Real-time Streaming Service
Handles WebSocket streaming for live dashboard
"""
import numpy as np
import asyncio
from typing import Dict, List, Optional
from core.audio_processor import AudioProcessor
from core.emotion_detector import EmotionDetector
from core.mood_mapper import MoodMapper
from core.metrics_processor import MetricsProcessor
from core.cluster_analyzer import ClusterAnalyzer
from core.insights_generator import InsightsGenerator
from core.risk_assessor import RiskAssessor
from config import settings


class RealtimeStreamingService:
    """
    Service for real-time emotion streaming
    Optimized for low-latency WebSocket updates with progressive streaming
    """
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.emotion_detector = EmotionDetector()
        self.mood_mapper = MoodMapper()
        self.metrics_processor = MetricsProcessor()
        self.cluster_analyzer = ClusterAnalyzer()
        self.insights_generator = InsightsGenerator()
        self.risk_assessor = RiskAssessor()
    
    def estimate_participants_from_audio(self, frames: List[np.ndarray], sample_rate: int) -> int:
        """
        Estimate number of participants from audio patterns
        Uses energy patterns and spectral analysis to simulate speaker diarization
        """
        if frames is None or len(frames) == 0:
            return 15  # Default fallback
        
        # Extract energy patterns
        energies = []
        for frame in frames:
            if len(frame) > 0:
                energy = np.sqrt(np.mean(frame ** 2)) * 100
                energies.append(energy)
        
        if len(energies) == 0:
            return 15
        
        # Analyze energy distribution to estimate speakers
        # More varied energy = more speakers
        energy_std = np.std(energies)
        energy_mean = np.mean(energies)
        
        # Count distinct energy levels (approximate speaker segments)
        # Higher variance and more energy spikes suggest more participants
        active_frames = [e for e in energies if e > 20]  # Filter out silence
        
        if len(active_frames) > 0:
            # Estimate based on energy variance and active speaking time
            variance_factor = min(energy_std / 10, 3.0)  # Cap at 3x
            active_ratio = len(active_frames) / len(energies)
            
            # Base estimate: 10-45 participants based on audio characteristics
            base_count = 15
            variance_contribution = int(variance_factor * 8)  # Up to +24
            activity_contribution = int(active_ratio * 12)    # Up to +12
            
            estimated_count = base_count + variance_contribution + activity_contribution
            
            # Clamp to realistic meeting size (8-50)
            estimated_count = max(8, min(50, estimated_count))
        else:
            estimated_count = 10  # Very quiet meeting
        
        print(f"🎤 Estimated {estimated_count} participants from audio analysis")
        return estimated_count
    
    async def initialize_stream(
        self,
        file_path: str,
        callback: Optional[callable] = None,
        initial_batch_duration: float = 5.0
    ) -> Dict:
        """
        Pre-process file for streaming with progressive loading
        
        Args:
            file_path: Path to video file
            callback: Optional callback to send partial data (for progressive streaming)
            initial_batch_duration: Duration in seconds for first batch (default 5s)
        
        Returns:
            Complete stream_data dict (may be partially populated initially)
        """
        print("🎬 Initializing real-time stream...")
        
        # Process audio
        audio_data = self.audio_processor.process_file(file_path)
        
        frames = audio_data['frames']
        timestamps = audio_data['timestamps']
        sample_rate = audio_data['sample_rate']
        duration = audio_data['duration']
        
        # Estimate participants from audio analysis
        participant_count = self.estimate_participants_from_audio(frames, sample_rate)
        
        total_frames = len(frames)
        
        # Calculate initial batch size (approximately initial_batch_duration seconds worth of frames)
        initial_batch_size = int((initial_batch_duration / duration) * total_frames)
        initial_batch_size = max(1, min(initial_batch_size, total_frames))
        
        print(f"📊 Total: {total_frames} frames ({duration:.1f}s)")
        print(f"🚀 Processing initial batch: {initial_batch_size} frames (~{initial_batch_duration:.1f}s)")
        
        # Process initial batch
        initial_frames = frames[:initial_batch_size]
        initial_emotions = self.emotion_detector.batch_analyze(
            initial_frames,
            sample_rate,
            use_parallel=True
        )
        
        # Calculate initial energy and categories
        initial_energy = self.metrics_processor.calculate_energy_timeline(initial_frames)
        initial_distribution, initial_categories = self.mood_mapper.get_category_distribution(
            initial_emotions,
            initial_energy
        )
        initial_category_displays = [
            self.mood_mapper.get_category_display(cat)
            for cat in initial_categories
        ]
        
        # Create initial stream data
        stream_data = {
            'duration': duration,
            'timestamps': timestamps,
            'energy_timeline': np.zeros(total_frames),  # Placeholder
            'emotion_series': [None] * total_frames,  # Placeholder
            'categories': [''] * total_frames,  # Placeholder
            'sample_rate': sample_rate,
            'is_fully_processed': False,
            'participant_count': participant_count  # Audio-based participant estimation
        }
        
        # Fill in initial batch data
        stream_data['energy_timeline'][:initial_batch_size] = initial_energy
        stream_data['emotion_series'][:initial_batch_size] = initial_emotions
        stream_data['categories'][:initial_batch_size] = initial_category_displays
        
        print(f"✅ Initial batch ready! Sending early 'ready' signal...")
        
        # If there are more frames, schedule background processing
        if initial_batch_size < total_frames:
            # Start background processing (non-blocking)
            asyncio.create_task(
                self._process_remaining_frames(
                    stream_data,
                    frames,
                    sample_rate,
                    initial_batch_size
                )
            )
            print(f"🔄 Background processing started for remaining {total_frames - initial_batch_size} frames...")
        else:
            stream_data['is_fully_processed'] = True
            print(f"✅ Stream fully processed!")
        
        return stream_data
    
    async def _process_remaining_frames(
        self,
        stream_data: Dict,
        frames: List[np.ndarray],
        sample_rate: int,
        start_idx: int
    ):
        """
        Process remaining frames in background
        Updates stream_data in-place
        """
        remaining_frames = frames[start_idx:]
        total_frames = len(frames)
        
        print(f"🔄 Background: Processing {len(remaining_frames)} remaining frames...")
        
        # Process in chunks to allow periodic yielding
        chunk_size = max(1, len(remaining_frames) // 4)  # Process in 4 chunks
        
        for chunk_start in range(0, len(remaining_frames), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(remaining_frames))
            chunk_frames = remaining_frames[chunk_start:chunk_end]
            
            # Analyze chunk
            chunk_emotions = self.emotion_detector.batch_analyze(
                chunk_frames,
                sample_rate,
                use_parallel=True
            )
            
            # Calculate energy and categories
            chunk_energy = self.metrics_processor.calculate_energy_timeline(chunk_frames)
            chunk_distribution, chunk_categories = self.mood_mapper.get_category_distribution(
                chunk_emotions,
                chunk_energy
            )
            chunk_category_displays = [
                self.mood_mapper.get_category_display(cat)
                for cat in chunk_categories
            ]
            
            # Update stream_data in-place
            actual_start = start_idx + chunk_start
            actual_end = actual_start + len(chunk_frames)
            
            stream_data['energy_timeline'][actual_start:actual_end] = chunk_energy
            stream_data['emotion_series'][actual_start:actual_end] = chunk_emotions
            stream_data['categories'][actual_start:actual_end] = chunk_category_displays
            
            progress = ((actual_end / total_frames) * 100)
            print(f"  Background progress: {actual_end}/{total_frames} frames ({progress:.0f}%)")
            
            # Yield control to allow other async operations
            await asyncio.sleep(0.1)
        
        stream_data['is_fully_processed'] = True
        print(f"✅ Background processing complete! Stream fully ready.")
    
    def get_realtime_data(
        self,
        stream_data: Dict,
        current_time: float
    ) -> Dict:
        """
        Get emotion data for current playback time
        Returns real-time KPIs for dashboard
        Handles partially processed data gracefully
        """
        timestamps = stream_data['timestamps']
        energy_timeline = stream_data['energy_timeline']
        categories = stream_data['categories']
        emotion_series = stream_data['emotion_series']
        
        # Find current index
        current_idx = self._find_nearest_index(timestamps, current_time)
        
        if current_idx < 0:
            return self._empty_update(current_time)
        
        # Check if this frame has been processed yet
        if emotion_series[current_idx] is None:
            # Frame not processed yet, return placeholder
            return {
                'time': current_time,
                'current_emotion': 'Processing...',
                'current_energy': 0.0,
                'avg_energy': 0.0,
                'silence_percentage': 0.0,
                'emotion_shifts': 0,
                'volatility': 0.0,
                'emotion_distribution': {},
                'timeline_length': 0,
                'is_processed': False
            }
        
        # Get data up to current time (only processed frames)
        energy_slice = energy_timeline[:current_idx + 1]
        category_slice = categories[:current_idx + 1]
        
        # Current values
        current_emotion = categories[current_idx]
        current_energy = energy_timeline[current_idx]
        
        # Convert NumPy array to list to avoid ambiguous truth value errors
        if isinstance(energy_slice, np.ndarray):
            energy_slice = energy_slice.tolist()
        
        # Calculate real-time metrics
        metrics = self.metrics_processor.calculate_realtime_metrics(
            energy_slice,
            category_slice,
            current_time
        )
        
        # Calculate participation percentage (based on non-silent frames)
        # Participation = (time speaking / total time) * 100
        non_silent_frames = sum(1 for e in energy_slice if e > 2.0)  # Energy > 2.0 indicates active speaking
        participation = (non_silent_frames / len(energy_slice)) * 100 if len(energy_slice) > 0 else 0.0
        
        # Calculate distribution up to now
        distribution = self._calculate_distribution(category_slice)
        
        return {
            'time': current_time,
            'current_emotion': current_emotion,
            'current_energy': float(current_energy),
            'avg_energy': float(metrics['avg_energy']),
            'silence_percentage': float(metrics['silence_pct']),
            'emotion_shifts': int(metrics['emotion_shifts']),
            'volatility': float(metrics['volatility']),
            'participation': float(participation),
            'emotion_distribution': distribution,
            'timeline_length': len(energy_slice),
            'is_processed': True,
            'participant_count': stream_data.get('participant_count', 0)
        }
    
    def _find_nearest_index(self, timestamps: np.ndarray, target_time: float) -> int:
        """Find nearest timestamp index"""
        if target_time < 0:
            return 0
        
        # Binary search for efficiency
        idx = np.searchsorted(timestamps, target_time)
        
        if idx >= len(timestamps):
            return len(timestamps) - 1
        
        return int(idx)
    
    def _calculate_distribution(self, categories: List[str]) -> Dict[str, float]:
        """Calculate category distribution"""
        if len(categories) == 0:
            return {}
        
        total = len(categories)
        distribution = {}
        
        for category in set(categories):
            count = categories.count(category)
            distribution[category] = (count / total) * 100
        
        return distribution
    
    def _empty_update(self, current_time: float) -> Dict:
        """Return empty update for invalid time"""
        return {
            'time': current_time,
            'current_emotion': 'N/A',
            'current_energy': 0.0,
            'avg_energy': 0.0,
            'silence_percentage': 0.0,
            'emotion_shifts': 0,
            'volatility': 0.0,
            'emotion_distribution': {},
            'timeline_length': 0
        }
    
    async def build_analysis_from_stream(self, stream_data: Dict, file_path: str) -> Dict:
        """
        Build full analysis from completed stream_data (no reprocessing needed)
        This is much faster than running analyze_full() again
        """
        print("🚀 Building analysis from stream_data...")
        
        timestamps = stream_data['timestamps']
        energy_timeline = stream_data['energy_timeline']
        emotion_series = stream_data['emotion_series']
        categories = stream_data['categories']
        duration = stream_data['duration']
        
        # Convert to list if numpy array
        if not isinstance(energy_timeline, list):
            energy_timeline = energy_timeline.tolist()
        
        # Step 1: Calculate metrics from energy timeline and categories
        # Calculate silence (frames below 20 energy)
        silent_count = sum(1 for e in energy_timeline if e < 20)
        silence_pct = (silent_count / len(energy_timeline)) * 100
        
        # Calculate participation (frames above 20 energy)
        participation = 100 - silence_pct
        
        # Average energy
        avg_energy = float(np.mean(energy_timeline))
        
        # Volatility
        volatility = self.metrics_processor.calculate_volatility(energy_timeline)
        
        # Emotion shifts
        emotion_shifts = self.metrics_processor.calculate_emotion_shifts(categories)
        
        metrics = {
            'energy_timeline': energy_timeline,
            'silence_percentage': silence_pct,
            'participation': participation,
            'avg_energy': avg_energy,
            'volatility': volatility,
            'emotion_shifts': emotion_shifts
        }
        
        # Step 2: Get distribution (use display names from categories)
        distribution = self._calculate_distribution(categories)
        dominant_emotion = self.mood_mapper.get_dominant_emotion(distribution)
        
        # Step 3: Clustering
        cluster_data = self.cluster_analyzer.analyze(
            emotion_series,
            energy_timeline
        )
        
        # Step 4: Risk assessment
        psych_risk = self.risk_assessor.assess_psychological_safety(
            metrics,
            distribution
        )
        
        # Step 5: Build timeline
        timeline_data = []
        for i, (time, energy, category, emotion) in enumerate(
            zip(timestamps, energy_timeline, categories, emotion_series)
        ):
            timeline_data.append({
                'time': float(time),
                'energy': float(energy),
                'category': category,
                'emotion_raw': emotion
            })
        
        # Step 6: Create summary
        summary = {
            'dominant_emotion': dominant_emotion,
            'avg_energy': float(metrics['avg_energy']),
            'silence_pct': float(metrics['silence_percentage']),
            'participation': float(metrics['participation']),
            'volatility': float(metrics['volatility']),
            'psych_risk': psych_risk,
            'distribution': distribution
        }
        
        # Step 7: Generate AI insights
        suggestions = self.insights_generator.generate_suggestions(summary)
        
        print("✅ Analysis built from stream_data (no reprocessing)!")
        
        return {
            'duration': float(duration),
            'summary': summary,
            'timeline': timeline_data,
            'clusters': cluster_data,
            'suggestions': suggestions
        }
