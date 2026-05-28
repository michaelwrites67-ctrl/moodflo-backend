"""
Report Generation Module
Generate PDF reports from analysis results using ReportLab
"""
from datetime import datetime
from typing import Dict, List
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER


class ReportGenerator:
    """Generate downloadable PDF reports from analysis results"""
    
    def __init__(self, results: Dict, session_id: str):
        self.results = results
        self.session_id = session_id
        self.timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    
    def strip_emoji(self, text: str) -> str:
        """Remove emoji characters from text for PDF compatibility"""
        import re
        # Remove emojis and extra spaces
        emoji_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emoticons
                                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                  u"\U00002702-\U000027B0"
                                  u"\U000024C2-\U0001F251"
                                  "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', text).strip()
    
    def generate_pdf_report(self) -> io.BytesIO:
        """Generate a professional PDF report for HR managers and team leaders"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        story = []
        
        summary = self.results['summary']
        timeline = self.results['timeline']
        duration = self.results['duration']
        suggestions = self.results.get('suggestions', 'No insights available.')
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1a1f2e'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1a1f2e'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            leading=16,
            textColor=colors.HexColor('#4a5568')
        )
        
        # Title & Logo placeholder
        story.append(Paragraph("Moodflo", title_style))
        story.append(Paragraph("Meeting Emotion Analysis Report", 
                              ParagraphStyle('Subtitle', parent=styles['Heading3'], 
                                           fontSize=14, textColor=colors.HexColor('#60a5fa'),
                                           alignment=TA_CENTER, spaceAfter=30)))
        story.append(Spacer(1, 0.3*inch))
        
        # Report Info
        info_data = [
            ["Report Generated:", self.timestamp],
            ["Session ID:", self.session_id[:13] + "..."],
            ["Meeting Duration:", f"{self.format_time(duration)} | Elapsed time {self.format_time(duration)}"],
        ]
        
        info_table = Table(info_data, colWidths=[1.8*inch, 4.2*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a1f2e')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#4a5568')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Key Metrics
        story.append(Paragraph("Overall Analysis - Key Metrics", heading_style))
        
        summary_data = [
            ["Metric", "Value", "Scale"],
            ["Dominant Emotion", self.strip_emoji(summary['dominant_emotion']), "Scale: 2 (from 5)"],
            ["Average Energy", f"{summary['avg_energy']:.1f} / 10", "0-10 Energy Levels"],
            ["Volatility Score", f"{int((summary['volatility'] or 0.7) * 100)}%", "Emotional Stability"],
            ["Engagement Level", f"{int((summary['participation'] or 54) * 0.8 + (summary['avg_energy'] or 38) * 0.2)} / 100", "Overall Team Engagement"],
            ["Room Sentiment", f"{int((summary['avg_energy'] or 38) + (summary['participation'] or 54)) / 2} / 100", "Collective Team Sentiment"],
            ["Psychological Safety", summary.get('psych_risk', 'Low'), "Risk Assessment"],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.2*inch, 2*inch, 2.3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#252b3b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Meeting Summary
        story.append(Paragraph("Meeting Summary - Emotional & Energetic Analysis", heading_style))
        
        # Use AI-generated summary if available, otherwise use fallback
        summary_content = self.results.get('ai_summary', suggestions)
        
        for paragraph in summary_content.split('\n\n'):
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), normal_style))
                story.append(Spacer(1, 0.12*inch))
        
        story.append(PageBreak())
        
        # Next Steps
        story.append(Paragraph("Team Wellbeing - What We Learned (and What to Do Next)", heading_style))
        
        # Use AI-generated next steps if available
        if 'ai_next_steps' in self.results and self.results['ai_next_steps']:
            # Parse AI-generated next steps
            ai_steps = self.results['ai_next_steps']
            for step in ai_steps.split('\n\n'):
                if step.strip():
                    # Parse step format: "1. Title. Description"
                    step_match = step.strip().split('. ', 1)
                    if len(step_match) >= 2:
                        number_title = step_match[0] + '.'
                        description = step_match[1]
                        
                        story.append(Paragraph(f"<b>{number_title}</b>", 
                                             ParagraphStyle('StepTitle', parent=normal_style, 
                                                          fontSize=11, textColor=colors.HexColor('#1a1f2e'),
                                                          spaceAfter=6, fontName='Helvetica-Bold')))
                        story.append(Paragraph(description, normal_style))
                        story.append(Spacer(1, 0.15*inch))
        else:
            # Fallback to default next steps
            next_steps = [
                ("1. Re-energise the room", 
                 "Energy dipped early and never recovered — add short energisers, rotate voices, or break long monologues. Small shifts = big lift."),
                ("2. Pull quieter people in", 
                 "Silence was high. Directly invite two or three quieter members to contribute. Spotlighting voices increases psychological safety."),
                ("3. Reduce cognitive fatigue", 
                 "The team showed steadiness but not engagement. Tighten agenda, shorten segments, and cut low-value talk time. Clarity = calm."),
                ("4. Celebrate micro-wins", 
                 "Volatility is stable, but mood is flat. Acknowledge small successes mid-meeting. It boosts momentum and emotional presence."),
                ("5. Add human check-ins", 
                 "Quick 'one-word check-in' at the start. Helps the team feel seen and levels the emotional playing field."),
                ("6. Break the monotony", 
                 "Flat emotion signals boredom or passive listening. Add visuals, vary speaker rhythm, or switch to collaborative tool use."),
                ("7. Encourage psychological presence", 
                 "Camera-off numbers didn't crash stability, but they hurt connection. Ask for cameras on during key decisions only."),
                ("8. Take the team out for a long lunch", 
                 "Reset morale. People talk more honestly over food than on slides."),
            ]
            
            for title, description in next_steps:
                story.append(Paragraph(f"<b>{title}</b>", 
                                     ParagraphStyle('StepTitle', parent=normal_style, 
                                                  fontSize=11, textColor=colors.HexColor('#1a1f2e'),
                                                  spaceAfter=6, fontName='Helvetica-Bold')))
                story.append(Paragraph(description, normal_style))
                story.append(Spacer(1, 0.15*inch))
        
        story.append(PageBreak())
        
        # Emotion Distribution
        story.append(Paragraph("Emotion Distribution Breakdown", heading_style))
        
        emotion_data = [["Emotion Category", "Percentage", "Interpretation"]]
        interpretations = {
            "Energised": "High energy, positive engagement",
            "Stressed": "Tension detected, possible pressure points",
            "Flat": "Low energy, possible disengagement",
            "Thoughtful": "Calm, focused discussion",
            "Volatile": "Unpredictable emotional patterns"
        }
        
        for emotion, percentage in summary['distribution'].items():
            clean_emotion = self.strip_emoji(emotion)
            interpretation = ""
            for key, val in interpretations.items():
                if key in clean_emotion:
                    interpretation = val
                    break
            # Format percentage to match interface display
            emotion_data.append([clean_emotion, f"{percentage:.1f}%", interpretation or "Standard emotional pattern"])
        
        emotion_table = Table(emotion_data, colWidths=[2.2*inch, 1.3*inch, 3*inch])
        emotion_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#252b3b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        story.append(emotion_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Timeline Summary
        story.append(Paragraph("Complete Emotion Signal - Timeline", heading_style))
        story.append(Paragraph("90-second interval snapshots showing energy levels throughout the meeting (mm:ss format)", 
                              ParagraphStyle('Caption', parent=normal_style, fontSize=9, 
                                           textColor=colors.HexColor('#6b7280'), spaceAfter=12)))
        
        timeline_data = [["Time (mm:ss)", "Emotion Category", "Energy Level"]]
        for t_idx in range(0, len(timeline), max(1, int(90 / 5))):
            point = timeline[t_idx]
            # Format time to match interface mm:ss display
            total_mins = int(point['time'] // 60)
            mins = total_mins % 60
            secs = int(point['time'] % 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            emotion = self.strip_emoji(point['category'])
            energy = f"{point['energy']:.1f}/10"
            timeline_data.append([time_str, emotion, energy])
        
        timeline_table = Table(timeline_data, colWidths=[1.2*inch, 3.5*inch, 1.8*inch])
        timeline_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#252b3b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        story.append(timeline_table)
        
        # Footer
        story.append(Spacer(1, 0.6*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#9ca3af'),
            alignment=TA_CENTER,
            spaceAfter=4
        )
        story.append(Paragraph(
            "This report is confidential and intended for HR managers, team leaders, and organizational development professionals.",
            footer_style
        ))
        story.append(Paragraph(
            "Privacy Protected: Only voice tone analyzed. No content recorded or stored.",
            footer_style
        ))
        story.append(Paragraph(
            "Generated by Moodflo v2.0 | © 2025",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_json_report(self) -> Dict:
        """Generate JSON export of all analysis data"""
        return {
            "metadata": {
                "session_id": self.session_id,
                "generated": self.timestamp,
                "duration": self.results['duration']
            },
            "summary": self.results['summary'],
            "timeline": self.results['timeline'],
            "clusters": self.results.get('clusters', {}),
            "suggestions": self.results.get('suggestions', ''),
            "ai_summary": self.results.get('ai_summary', ''),
            "ai_next_steps": self.results.get('ai_next_steps', ''),
            "version": "2.0.0"
        }

