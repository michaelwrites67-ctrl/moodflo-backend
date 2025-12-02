"""
AI Insights Generator Module  
Generates actionable suggestions using OpenAI GPT-4
"""
from openai import OpenAI
from typing import Dict
from config import settings


class InsightsGenerator:
    """Generate AI-powered meeting insights"""
    
    def __init__(self, api_key: str = None):
        """Initialize with optional API key"""
        self.api_key = api_key or settings.OPENAI_API_KEY
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self.client = None
        else:
            self.client = None
    
    def generate_suggestions(self, analysis_data: Dict) -> str:
        """
        Generate actionable suggestions based on analysis
        Falls back to rule-based if API key not available
        """
        if not self.client or not self.api_key:
            return self._fallback_suggestions(analysis_data)
        
        try:
            prompt = self._build_prompt(analysis_data)
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert meeting coach analyzing emotional patterns from acoustic analysis only (no content). Provide 4-5 concise, actionable suggestions focused on psychological safety and practical next steps."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"OpenAI error: {e}, using fallback")
            return self._fallback_suggestions(analysis_data)
    
    def _build_prompt(self, data: Dict) -> str:
        """Build prompt for GPT-4"""
        prompt = f"""Meeting Acoustic Analysis Summary:

Dominant Emotion: {data['dominant_emotion']}
Average Energy Level: {data['avg_energy']:.1f}/100
Silence Percentage: {data['silence_pct']:.1f}%
Participation Rate: {data['participation']:.1f}%
Volatility Score: {data['volatility']:.1f}/10
Psychological Safety Risk: {data['psych_risk']}

Emotion Distribution:
"""
        for emotion, percentage in data['distribution'].items():
            prompt += f"- {emotion}: {percentage:.1f}%\n"
        
        prompt += "\nGenerate 4-5 specific, actionable suggestions for the meeting leader based on these acoustic patterns."
        return prompt
    
    def _fallback_suggestions(self, data: Dict) -> str:
        """
        Rule-based suggestions when OpenAI unavailable
        Based on dominant emotion and risk level
        """
        suggestions = []
        dominant = data['dominant_emotion']
        
        # Category-specific suggestions
        if "Energised" in dominant:
            category_header = "⚡ ENERGISED MEETING\nTeam showed high energy and positive engagement.\n"
            suggestions.extend([
                "✓ Momentum is strong — protect it by ending meetings early this week",
                "✓ Share quick wins publicly to reward the positive energy",
                "✓ Add buffer time between meetings to prevent burnout",
                "✓ Capture key insights while engagement is at peak",
                "✓ Consider replicating this meeting format in future"
            ])
        
        elif "Stressed" in dominant or "Tense" in dominant:
            category_header = "🔥 STRESSED / TENSE MEETING\nTeam tone indicated stress and tension.\n"
            suggestions.extend([
                "⚠️ Cancel or postpone non-essential meetings this week",
                "⚠️ Offer one-to-one check-ins to understand concerns",
                "⚠️ Share something positive that's under control",
                "⚠️ Consider postponing major decisions until tension eases",
                "⚠️ Review workload distribution across the team"
            ])
        
        elif "Flat" in dominant or "Disengaged" in dominant:
            category_header = "🌫️ FLAT / DISENGAGED MEETING\nTeam showed low energy and engagement.\n"
            suggestions.extend([
                "⚡ Cut meeting time by 50% next week to respect energy levels",
                "⚡ Consider ending the week early for recovery",
                "⚡ Create space for anonymous feedback",
                "⚡ Introduce interactive elements or breakout discussions",
                "⚡ Review if meeting objectives are clear and relevant"
            ])
        
        elif "Thoughtful" in dominant or "Constructive" in dominant:
            category_header = "💬 THOUGHTFUL / FOCUSED MEETING\nTeam was calm, steady, and reflective.\n"
            suggestions.extend([
                "✓ Excellent meeting dynamics — maintain this format",
                "✓ Capture insights and decisions while they're fresh",
                "✓ Ask team: 'What helped today's flow?'",
                "✓ Document and repeat successful elements",
                "✓ Consider this a baseline for future meetings"
            ])
        
        elif "Volatile" in dominant or "Unstable" in dominant:
            category_header = "🌪️ VOLATILE / UNSTABLE MEETING\nEmotional tone was unpredictable and mixed.\n"
            suggestions.extend([
                "⚠️ Follow up individually with less active participants",
                "⚠️ Reiterate shared goals and objectives in writing",
                "⚠️ Consider bringing in facilitation support",
                "⚠️ Break large group into smaller discussion groups",
                "⚠️ Review meeting structure and participation balance"
            ])
        
        else:
            category_header = "MEETING ANALYSIS\n"
            suggestions.extend([
                "Review meeting structure and participation patterns",
                "Consider individual check-ins with team members",
                "Monitor emotional patterns in upcoming meetings",
                "Gather feedback on meeting effectiveness"
            ])
        
        # Add psychological safety context
        risk_level = data['psych_risk']
        
        if risk_level == "High":
            psych_section = f"""

🧠 PSYCHOLOGICAL SAFETY RISK: HIGH
Critical factors detected — immediate action required.

Metrics:
• Silence: {data['silence_pct']:.1f}%
• Stress: {data['distribution'].get('🔥 Stressed/Tense', 0):.1f}%
• Volatility: {data['volatility']:.1f}

URGENT ACTIONS:
• Pause all group decision-making immediately
• Score team's current working experience (1-5 scale)
• Run psychological safety retrospective
• Schedule one-to-one's with all participants
• Address concerns before proceeding with regular schedule
"""
            return category_header + "\nRECOMMENDATIONS:\n" + "\n".join(suggestions[:5]) + psych_section
        
        elif risk_level == "Medium":
            psych_note = f"""

⚠️ PSYCHOLOGICAL SAFETY RISK: MEDIUM
Some warning signs detected — monitor closely.

Metrics:
• Silence: {data['silence_pct']:.1f}%
• Stress: {data['distribution'].get('🔥 Stressed/Tense', 0):.1f}%
• Volatility: {data['volatility']:.1f}

NEXT STEPS:
• Monitor team dynamics in next session
• Create anonymous feedback channel
• Check in with quieter team members
"""
            return category_header + "\nRECOMMENDATIONS:\n" + "\n".join(suggestions[:5]) + psych_note
        
        else:
            return category_header + "\nRECOMMENDATIONS:\n" + "\n".join(suggestions[:5]) + "\n\n✓ Psychological Safety: LOW RISK — Team dynamics appear healthy"
    
    def generate_next_steps(self, summary_data: Dict) -> str:
        """Generate next steps for meeting improvement"""
        if not self.client or not self.api_key:
            return self._fallback_next_steps(summary_data)
        
        try:
            prompt = f"""Based on this meeting analysis, generate 8 specific, actionable next steps for team wellbeing and meeting improvement:

Dominant Emotion: {summary_data.get('dominant_emotion', 'Neutral')}
Average Energy: {summary_data.get('avg_energy', 50):.1f}/100
Silence: {summary_data.get('silence_pct', 30):.1f}%
Participation: {summary_data.get('participation', 60):.1f}%
Volatility: {summary_data.get('volatility', 5):.1f}/10

Format as: '1. Title. Brief explanation (1-2 sentences).'
Focus on practical team wellness and meeting dynamics."""
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert meeting facilitator focused on team wellbeing. Provide practical, actionable next steps."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"OpenAI error: {e}, using fallback next steps")
            return self._fallback_next_steps(summary_data)
    
    def _fallback_next_steps(self, data: Dict) -> str:
        """Generate fallback next steps when OpenAI unavailable"""
        steps = [
            "1. Re-energise the room. Energy dipped early and never recovered — add short energisers, rotate voices, or break long monologues. Small shifts = big lift.",
            "2. Pull quieter people in. Silence was high. Directly invite two or three quieter members to contribute. Spotlighting voices increases psychological safety.",
            "3. Reduce cognitive fatigue. The team showed steadiness but not engagement. Tighten agenda, shorten segments, and cut low-value talk time. Clarity = calm.",
            "4. Celebrate micro-wins. Volatility is stable, but mood is flat. Acknowledge small successes mid-meeting. It boosts momentum and emotional presence.",
            "5. Add human check-ins. Quick \"one-word check-in\" at the start. Helps the team feel seen and levels the emotional playing field.",
            "6. Break the monotony. Flat emotion signals boredom or passive listening. Add visuals, vary speaker rhythm, or switch to collaborative tool use.",
            "7. Encourage psychological presence. Camera-off numbers didn't crash stability, but they hurt connection. Ask for cameras on during key decisions only — lowers pressure, increases visibility.",
            "8. Take the team out for a long lunch. Reset morale. People talk more honestly over food than on slides."
        ]
        
        return "\n\n".join(steps)
    
    def generate_summary(self, summary_data: Dict) -> str:
        """Generate AI-powered meeting summary"""
        if not self.client or not self.api_key:
            return self._fallback_summary(summary_data)
        
        try:
            prompt = f"""Based on this meeting's emotional and behavioral analysis, generate a comprehensive meeting summary in 3-4 paragraphs:

Dominant Emotion: {summary_data.get('dominant_emotion', 'Neutral')}
Average Energy: {summary_data.get('avg_energy', 50):.1f}/100
Participation: {summary_data.get('participation', 60):.1f}%
Volatility: {summary_data.get('volatility', 5):.1f}/10
Psychological Safety Risk: {summary_data.get('psych_risk', 'Low')}

Focus on:
- Overall emotional tone and energy patterns
- Team engagement and participation dynamics
- Psychological safety and group cohesion
- Key observations about meeting effectiveness

Write in a professional, analytical tone for team leaders."""
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert meeting analyst who provides insights into team dynamics and emotional patterns. Write clear, professional summaries that help team leaders understand their meeting effectiveness."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"OpenAI error: {e}, using fallback summary")
            return self._fallback_summary(summary_data)
    
    def _fallback_summary(self, data: Dict) -> str:
        """Generate fallback summary when OpenAI unavailable"""
        dominant_emotion = data.get('dominant_emotion', 'Neutral')
        avg_energy = data.get('avg_energy', 50)
        participation = data.get('participation', 60)
        volatility = data.get('volatility', 5)
        
        summary = f"""The meeting displayed a {dominant_emotion.lower()} emotional tone throughout the session, with energy levels averaging {avg_energy:.1f} out of 100. The team demonstrated {participation:.1f}% active participation, indicating {'strong engagement' if participation > 70 else 'moderate engagement' if participation > 50 else 'limited engagement'} across participants.

Emotional stability remained {'consistent' if volatility < 4 else 'somewhat variable' if volatility < 7 else 'highly variable'} with a volatility score of {volatility:.1f}/10. This suggests the team operated {'smoothly with minimal emotional disruptions' if volatility < 4 else 'with some fluctuations in group dynamics' if volatility < 7 else 'through significant emotional shifts that may have impacted focus'}.

From a psychological safety perspective, the meeting showed {'positive indicators' if data.get('psych_risk', 'Low') == 'Low' else 'some concerns' if data.get('psych_risk', 'Medium') == 'Medium' else 'significant warning signs'} of team cohesion and open communication. {'The environment appeared conducive to productive collaboration' if data.get('psych_risk', 'Low') == 'Low' else 'There may be opportunities to improve team comfort and participation' if data.get('psych_risk', 'Medium') == 'Medium' else 'Immediate attention to team dynamics and communication patterns is recommended'}.

Overall, this meeting represents {'an effective team session' if avg_energy > 60 and participation > 60 else 'a functional but improvable team interaction' if avg_energy > 40 or participation > 40 else 'a session that would benefit from significant process adjustments'} with clear patterns that can inform future meeting optimization and team development strategies."""
        
        return summary
