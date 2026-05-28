"""
Risk Assessment Module
Evaluates psychological safety risks based on meeting metrics
"""
from typing import Dict, List
from config import settings


class RiskAssessor:
    """Assess psychological safety risks"""
    
    @staticmethod
    def assess_psychological_safety(
        metrics: Dict,
        distribution: Dict[str, float]
    ) -> str:
        """
        Assess psychological safety risk level
        
        Risk factors:
        - High silence percentage
        - High stress levels
        - High volatility
        - Low participation
        
        Returns: "Low", "Medium", or "High"
        """
        silence_pct = metrics.get('silence_percentage', 0)
        volatility = metrics.get('volatility', 0)
        participation = metrics.get('participation', 100)
        
        # Get stress percentage from distribution
        stress_pct = 0
        for key, value in distribution.items():
            if 'Stressed' in key or 'Tense' in key:
                stress_pct = value
                break
        
        # Check high risk thresholds
        high_thresholds = settings.PSYCH_SAFETY_THRESHOLDS['high_risk']
        
        high_risk_count = 0
        if silence_pct > high_thresholds['silence']:
            high_risk_count += 1
        if stress_pct > high_thresholds['stress']:
            high_risk_count += 1
        if volatility > high_thresholds['volatility']:
            high_risk_count += 1
        
        # High risk if 2+ factors
        if high_risk_count >= 2:
            return "High"
        
        # Check medium risk thresholds
        medium_thresholds = settings.PSYCH_SAFETY_THRESHOLDS['medium_risk']
        
        medium_risk_count = 0
        if silence_pct > medium_thresholds['silence']:
            medium_risk_count += 1
        if stress_pct > medium_thresholds['stress']:
            medium_risk_count += 1
        if volatility > medium_thresholds['volatility']:
            medium_risk_count += 1
        if participation < 50:
            medium_risk_count += 1
        
        # Medium risk if 2+ factors
        if medium_risk_count >= 2:
            return "Medium"
        
        return "Low"
    
    @staticmethod
    def get_risk_recommendations(risk_level: str) -> List[str]:
        """Get specific recommendations based on risk level"""
        if risk_level == "High":
            return [
                "⚠️ URGENT: Pause all group decision-making",
                "Run one-to-one check-ins with all team members",
                "Consider psychological safety retrospective",
                "Address concerns before next meeting"
            ]
        elif risk_level == "Medium":
            return [
                "Monitor team dynamics closely",
                "Create anonymous feedback channel",
                "Check in with quieter team members",
                "Consider shorter, more focused meetings"
            ]
        else:
            return [
                "Current team dynamics appear healthy",
                "Maintain open communication channels",
                "Continue monitoring participation patterns"
            ]
