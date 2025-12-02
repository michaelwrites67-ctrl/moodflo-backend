"""
Mood Mapping Module
Maps raw emotions to Moodflo categories
"""
import numpy as np
from typing import Dict, List, Tuple
from config import settings


class MoodMapper:
    """Map Vokaturi emotions to Moodflo categories"""
    
    @staticmethod
    def map_emotion_to_category(emotion_dict: Dict[str, float], energy: float) -> str:
        """
        Map raw emotion probabilities to Moodflo category
        
        Categories:
        - energised: Happy + high energy
        - stressed: Angry/Fear + high energy
        - flat: Low energy + neutral/sad
        - thoughtful: Neutral + moderate energy
        - volatile: Mixed/unpredictable
        """
        happy = emotion_dict.get('happy', 0)
        angry = emotion_dict.get('angry', 0)
        fearful = emotion_dict.get('fearful', 0)
        sad = emotion_dict.get('sad', 0)
        neutral = emotion_dict.get('neutral', 0)
        
        # Energised: High happiness + good energy
        if happy > 0.4 and energy > 30:
            return "energised"
        
        # Stressed/Tense: Anger or fear dominant
        if (angry + fearful) > 0.35 or (energy > 40 and angry > 0.25):
            return "stressed"
        
        # Flat/Disengaged: Low energy
        if neutral > 0.55 and energy < 20:
            return "flat"
        
        # Thoughtful/Constructive: Calm and moderate
        if neutral > 0.35 and 20 <= energy <= 45 and sad < 0.25:
            return "thoughtful"
        
        # Volatile: Everything else
        return "volatile"
    
    @staticmethod
    def get_category_display(category: str) -> str:
        """Get display name for category"""
        return settings.MOODFLO_CATEGORIES.get(category, category)
    
    @staticmethod
    def get_category_distribution(
        emotion_series: List[Dict[str, float]],
        energy_series: List[float]
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Calculate distribution of categories across timeline
        Returns: (distribution dict, category list)
        """
        categories = []
        
        for emotion, energy in zip(emotion_series, energy_series):
            category = MoodMapper.map_emotion_to_category(emotion, energy)
            categories.append(category)
        
        # Calculate percentages
        distribution = {}
        total = len(categories)
        
        for category in set(categories):
            count = categories.count(category)
            display_name = MoodMapper.get_category_display(category)
            distribution[display_name] = (count / total) * 100
        
        return distribution, categories
    
    @staticmethod
    def get_dominant_emotion(distribution: Dict[str, float]) -> str:
        """Get the dominant emotion from distribution"""
        if len(distribution) == 0:
            return settings.MOODFLO_CATEGORIES["thoughtful"]
        return max(distribution.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def get_category_color(category: str) -> str:
        """Get color code for category visualization"""
        color_map = {
            "energised": "#00d4aa",
            "stressed": "#ff4444",
            "flat": "#888888",
            "thoughtful": "#667eea",
            "volatile": "#ffa500"
        }
        return color_map.get(category, "#667eea")
