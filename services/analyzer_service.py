"""
Analyzer Service
Handles comprehensive meeting analysis (Overall Analysis section)
"""
import pandas as pd
from typing import Dict
from core.audio_processor import AudioProcessor
from core.emotion_detector import EmotionDetector
from core.mood_mapper import MoodMapper
from core.metrics_processor import MetricsProcessor
from core.cluster_analyzer import ClusterAnalyzer
from core.risk_assessor import RiskAssessor
from core.insights_generator import InsightsGenerator
from config import settings


class AnalyzerService:
    """
    Service for comprehensive meeting analysis
    Used for "Overall Analysis" section
    """
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.emotion_detector = EmotionDetector()
        self.mood_mapper = MoodMapper()
        self.cluster_analyzer = ClusterAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.insights_generator = InsightsGenerator()
    
    async def analyze_full(self, file_path: str) -> Dict:
        """
        Run complete analysis on meeting recording
        Returns comprehensive results including clustering and AI insights
        """
        # Step 1: Process audio
        print("📊 Processing audio...")
        audio_data = self.audio_processor.process_file(file_path)
        
        frames = audio_data['frames']
        timestamps = audio_data['timestamps']
        sample_rate = audio_data['sample_rate']
        duration = audio_data['duration']
        
        # Step 2: Detect emotions (with parallel processing)
        print("🎭 Detecting emotions...")
        emotion_series = self.emotion_detector.batch_analyze(
            frames,
            sample_rate,
            use_parallel=True
        )
        
        # Step 3: Calculate metrics
        print("📈 Computing metrics...")
        metrics_proc = MetricsProcessor(sample_rate)
        metrics = metrics_proc.calculate_all_metrics(
            frames,
            emotion_series,
            audio_data.get('full_audio')
        )
        
        # Step 4: Map to categories
        print("🎯 Mapping emotions...")
        distribution, categories = self.mood_mapper.get_category_distribution(
            emotion_series,
            metrics['energy_timeline']
        )
        
        dominant_emotion = self.mood_mapper.get_dominant_emotion(distribution)
        
        # Step 5: Cluster analysis (for Overall Analysis only)
        print("🔬 Analyzing patterns...")
        cluster_data = self.cluster_analyzer.analyze(
            emotion_series,
            metrics['energy_timeline']
        )
        
        # Step 6: Risk assessment
        print("⚠️ Assessing risks...")
        psych_risk = self.risk_assessor.assess_psychological_safety(
            metrics,
            distribution
        )
        
        # Step 7: Build timeline
        timeline_data = []
        for i, (time, energy, category, emotion) in enumerate(
            zip(timestamps, metrics['energy_timeline'], categories, emotion_series)
        ):
            timeline_data.append({
                'time': float(time),
                'energy': float(energy),
                'category': self.mood_mapper.get_category_display(category),
                'emotion_raw': emotion
            })
        
        # Step 8: Create summary
        summary = {
            'dominant_emotion': dominant_emotion,
            'avg_energy': float(metrics['avg_energy']),
            'silence_pct': float(metrics['silence_percentage']),
            'participation': float(metrics['participation']),
            'volatility': float(metrics['volatility']),
            'psych_risk': psych_risk,
            'distribution': distribution
        }
        
        # Step 9: Generate AI insights
        print("💡 Generating insights...")
        suggestions = self.insights_generator.generate_suggestions(summary)
        
        print("✅ Analysis complete!")
        
        return {
            'duration': float(duration),
            'summary': summary,
            'timeline': timeline_data,
            'clusters': cluster_data,
            'suggestions': suggestions
        }
    
    def get_timeline_at_time(
        self,
        timeline_data: list,
        current_time: float
    ) -> list:
        """
        Get timeline data up to current time
        Used for progressive loading in UI
        """
        return [
            point for point in timeline_data
            if point['time'] <= current_time
        ]
