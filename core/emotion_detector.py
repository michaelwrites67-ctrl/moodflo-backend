"""
Emotion Detection Module
Uses Vokaturi SDK with fallback to acoustic analysis
"""
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from config import settings


# Load Vokaturi
vokaturi_lib_path = settings.VOKATURI_PATH / "api"
sys.path.insert(0, str(vokaturi_lib_path))

try:
    import Vokaturi
    VOKATURI_AVAILABLE = True
except ImportError:
    VOKATURI_AVAILABLE = False
    print("Warning: Vokaturi not available, using fallback analysis")


def _analyze_frame_worker(frame_data: tuple) -> Dict[str, float]:
    """
    Worker function for parallel processing
    Analyzes a single frame in a separate process
    """
    frame, sample_rate, lib_path = frame_data
    
    # Re-import Vokaturi in worker process
    try:
        if not hasattr(_analyze_frame_worker, 'vokaturi_loaded'):
            Vokaturi.load(str(lib_path))
            _analyze_frame_worker.vokaturi_loaded = True
    except Exception as e:
        return _fallback_analysis_static(frame)
    
    try:
        buffer_length = len(frame)
        c_buffer = Vokaturi.float64array(buffer_length)
        
        # Convert to float
        if frame.dtype == np.int16:
            c_buffer[:] = frame[:] / 32768.0
        elif frame.dtype == np.int32:
            c_buffer[:] = frame[:] / 2147483648.0
        else:
            c_buffer[:] = frame[:]
        
        # Create voice
        voice = Vokaturi.Voice(float(sample_rate), buffer_length, 0)
        voice.fill_float64array(buffer_length, c_buffer)
        
        # Extract emotions
        quality = Vokaturi.Quality()
        emotion = Vokaturi.EmotionProbabilities()
        voice.extract(quality, emotion)
        
        voice.destroy()
        
        if quality.valid:
            return {
                'neutral': emotion.neutrality,
                'happy': emotion.happiness,
                'sad': emotion.sadness,
                'angry': emotion.anger,
                'fearful': emotion.fear
            }
    except Exception:
        pass
    
    return _fallback_analysis_static(frame)


def _fallback_analysis_static(frame: np.ndarray) -> Dict[str, float]:
    """Static fallback analysis (can be pickled for multiprocessing)"""
    energy = float(np.sqrt(np.mean(frame ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))))
    
    if energy > 0.08:
        if zcr > 0.15:
            return {'neutral': 0.2, 'happy': 0.5, 'sad': 0.1, 'angry': 0.1, 'fearful': 0.1}
        else:
            return {'neutral': 0.2, 'happy': 0.1, 'sad': 0.1, 'angry': 0.4, 'fearful': 0.2}
    elif energy < 0.02:
        return {'neutral': 0.4, 'happy': 0.1, 'sad': 0.3, 'angry': 0.1, 'fearful': 0.1}
    else:
        return {'neutral': 0.6, 'happy': 0.1, 'sad': 0.1, 'angry': 0.1, 'fearful': 0.1}


class EmotionDetector:
    """Detect emotions from audio frames using Vokaturi or fallback"""
    
    def __init__(self):
        self.vokaturi_loaded = False
        
        if VOKATURI_AVAILABLE:
            lib_path = self._get_vokaturi_lib_path()
            if lib_path and os.path.exists(lib_path):
                try:
                    Vokaturi.load(str(lib_path))
                    self.vokaturi_loaded = True
                    print(f"✓ Vokaturi loaded from {lib_path}")
                except Exception as e:
                    print(f"Warning: Could not load Vokaturi: {e}")
    
    def _get_vokaturi_lib_path(self) -> Path:
        """Get platform-specific Vokaturi library path"""
        base_path = settings.VOKATURI_PATH / "lib" / "open"
        
        import struct
        if sys.platform == "win32":
            if struct.calcsize("P") == 4:
                lib_file = "win/OpenVokaturi-4-0-win32.dll"
            else:
                lib_file = "win/OpenVokaturi-4-0-win64.dll"
        elif sys.platform == "darwin":
            lib_file = "macos/OpenVokaturi-4-0-mac.dylib"
        else:
            lib_file = "linux/OpenVokaturi-4-0-linux.so"
        
        return base_path / lib_file
    
    def analyze_frame(self, frame: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Analyze single audio frame for emotions
        Returns: dict with emotion probabilities
        """
        if not self.vokaturi_loaded:
            return self._fallback_analysis(frame)
        
        try:
            buffer_length = len(frame)
            c_buffer = Vokaturi.float64array(buffer_length)
            
            # Convert to float
            if frame.dtype == np.int16:
                c_buffer[:] = frame[:] / 32768.0
            elif frame.dtype == np.int32:
                c_buffer[:] = frame[:] / 2147483648.0
            else:
                c_buffer[:] = frame[:]
            
            # Create voice (sample_rate, buffer_length, multi_threading)
            voice = Vokaturi.Voice(float(sample_rate), buffer_length, 0)
            voice.fill_float64array(buffer_length, c_buffer)
            
            # Extract emotions
            quality = Vokaturi.Quality()
            emotion = Vokaturi.EmotionProbabilities()
            voice.extract(quality, emotion)
            
            voice.destroy()
            
            if quality.valid:
                return {
                    'neutral': emotion.neutrality,
                    'happy': emotion.happiness,
                    'sad': emotion.sadness,
                    'angry': emotion.anger,
                    'fearful': emotion.fear
                }
        
        except Exception as e:
            print(f"Vokaturi error: {e}, using fallback")
        
        return self._fallback_analysis(frame)
    
    def _fallback_analysis(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Fallback emotion analysis using acoustic features
        When Vokaturi is unavailable
        """
        # Compute acoustic features
        energy = float(np.sqrt(np.mean(frame ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))))
        
        # Heuristic emotion mapping
        if energy > 0.08:
            if zcr > 0.15:
                # High energy + high variation = Happy/Energised
                return {
                    'neutral': 0.2,
                    'happy': 0.5,
                    'sad': 0.1,
                    'angry': 0.1,
                    'fearful': 0.1
                }
            else:
                # High energy + low variation = Angry/Stressed
                return {
                    'neutral': 0.2,
                    'happy': 0.1,
                    'sad': 0.1,
                    'angry': 0.4,
                    'fearful': 0.2
                }
        elif energy < 0.02:
            # Low energy = Sad/Flat
            return {
                'neutral': 0.4,
                'happy': 0.1,
                'sad': 0.3,
                'angry': 0.1,
                'fearful': 0.1
            }
        else:
            # Medium energy = Neutral
            return {
                'neutral': 0.6,
                'happy': 0.1,
                'sad': 0.1,
                'angry': 0.1,
                'fearful': 0.1
            }
    
    def batch_analyze(
        self,
        frames: List[np.ndarray],
        sample_rate: int,
        use_parallel: bool = True
    ) -> List[Dict[str, float]]:
        """
        Analyze multiple frames
        Uses ProcessPoolExecutor for true parallel processing
        """
        if not use_parallel or len(frames) < settings.BATCH_SIZE:
            # Sequential processing for small batches
            return [self.analyze_frame(frame, sample_rate) for frame in frames]
        
        # Parallel processing with ProcessPoolExecutor
        print(f"🚀 Starting parallel emotion detection for {len(frames)} frames...")
        results = [None] * len(frames)
        lib_path = self._get_vokaturi_lib_path()
        
        # Prepare frame data with index, frame, sample_rate, lib_path
        frame_data = [(frame, sample_rate, lib_path) for frame in frames]
        
        with ProcessPoolExecutor(max_workers=settings.PARALLEL_WORKERS) as executor:
            future_to_idx = {
                executor.submit(_analyze_frame_worker, data): idx
                for idx, data in enumerate(frame_data)
            }
            
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                
                # Progress logging every 10%
                if completed % max(1, len(frames) // 10) == 0:
                    print(f"  Progress: {completed}/{len(frames)} frames ({(completed/len(frames)*100):.0f}%)")
        
        print(f"✅ Parallel emotion detection complete!")
        return results
    
    def analyze_streaming(
        self,
        frame: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Analyze single frame for real-time streaming
        Optimized for low latency
        """
        return self.analyze_frame(frame, sample_rate)
