"""PDF report builder for end-of-meeting exports."""

from io import BytesIO
from typing import Any, Dict, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PdfReportService:
    """Builds a compact PDF report from finalized meeting metrics."""

    @staticmethod
    def _safe_text(value: Any, default: str = "-") -> str:
        text = str(value).strip() if value is not None else ""
        return text or default

    @staticmethod
    def _fmt_num(value: Any, suffix: str = "", decimals: int = 1) -> str:
        try:
            num = float(value)
            return f"{num:.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _dict_to_inline(metrics: Dict[str, Any], suffix: str = "%") -> str:
        if not metrics:
            return "-"

        parts = []
        for key, value in metrics.items():
            label = str(key).replace("_", " ").title()
            try:
                num = float(value)
                parts.append(f"{label}: {num:.1f}{suffix}")
            except (TypeError, ValueError):
                parts.append(f"{label}: {value}")
        return " | ".join(parts)

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
        cleaned = "_".join(part for part in cleaned.split("_") if part)
        return cleaned or "meeting"

    def build_filename(self, meeting_title: str, date_str: str) -> str:
        title_part = self._sanitize_filename(meeting_title)
        date_part = self._sanitize_filename(date_str)
        return f"moodflo_report_{title_part}_{date_part}.pdf"

    @staticmethod
    def _risk_palette(risk_level: str) -> Dict[str, str]:
        level = str(risk_level or "medium").strip().lower()
        if level == "low":
            return {
                "main": "#2f9e7d",
                "soft": "#e7f7f1",
                "text": "#1b5e4a",
            }
        if level == "high":
            return {
                "main": "#d9485f",
                "soft": "#fff0f2",
                "text": "#8b1f30",
            }
        return {
            "main": "#d68a2f",
            "soft": "#fff6e8",
            "text": "#875520",
        }

    def build_end_of_meeting_pdf(self, report: Dict[str, Any]) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=16 * mm,
            rightMargin=16 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title="Moodflo End-of-Meeting Report",
        )

        meeting_title = self._safe_text(report.get("meeting_title"), "Meeting")
        room_name = self._safe_text(report.get("room_name"), "N/A")
        date_text = self._safe_text(report.get("date"))
        start_text = self._safe_text(report.get("start_time"))
        end_text = self._safe_text(report.get("end_time"))
        duration_text = self._safe_text(report.get("total_duration"))
        risk_level = self._safe_text(report.get("risk_level"), "medium").lower()
        risk = self._risk_palette(risk_level)

        styles = getSampleStyleSheet()
        banner_title_style = ParagraphStyle(
            "BannerTitle",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.white,
        )
        banner_meta_style = ParagraphStyle(
            "BannerMeta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#d7e7f1"),
        )
        section_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1d3f55"),
            fontSize=12,
            leading=14,
            spaceBefore=8,
            spaceAfter=6,
        )
        card_style = ParagraphStyle(
            "MetricCard",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2b3e4c"),
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#253746"),
        )
        bullet_style = ParagraphStyle(
            "BulletText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.7,
            leading=14,
            textColor=colors.HexColor("#2b3d4b"),
        )
        note_style = ParagraphStyle(
            "Note",
            parent=styles["Italic"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#5f6f7d"),
            alignment=TA_CENTER,
        )

        story = []

        header_html = (
            "<b>Moodflo End-of-Meeting Report</b><br/>"
            "<font size='9'>Acoustic meeting dynamics summary (no transcript or content capture)</font><br/><br/>"
            f"<font size='11'><b>{meeting_title}</b></font><br/>"
            f"<font size='9'>Room: {room_name}   |   Date: {date_text}</font><br/>"
            f"<font size='9'>Start: {start_text}   |   End: {end_text}   |   Duration: {duration_text}</font>"
        )
        header_block = Table(
            [[Paragraph(header_html, banner_meta_style)]],
            colWidths=[doc.width],
        )
        header_block.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#174a68")),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#0f3449")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(header_block)
        story.append(Spacer(1, 10))

        risk_chip = Table(
            [
                [
                    Paragraph(
                        f"<font color='{risk['text']}'><b>Risk Level: {risk_level.upper()}</b></font>",
                        body_style,
                    )
                ]
            ],
            colWidths=[doc.width],
        )
        risk_chip.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(risk["soft"])),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(risk["main"])),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(risk_chip)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Final Acoustic Metrics", section_style))

        def metric_card(label: str, value: str) -> Paragraph:
            text = (
                f"<font color='#647785' size='8'>{label.upper()}</font><br/>"
                f"<font color='#1f3b4d' size='12'><b>{value}</b></font>"
            )
            return Paragraph(text, card_style)

        cards = [
            metric_card(
                "Final Team Tone", self._safe_text(report.get("final_team_tone"))
            ),
            metric_card(
                "Average Speaking Energy",
                self._fmt_num(report.get("average_speaking_energy"), "/100"),
            ),
            metric_card(
                "Tone Stability", self._fmt_num(report.get("tone_stability"), "%")
            ),
            metric_card(
                "Speaking Balance", self._safe_text(report.get("speaking_balance"))
            ),
            metric_card(
                "Silence", self._fmt_num(report.get("silence_percentage"), "%")
            ),
            metric_card("Room", room_name),
        ]

        metrics_cards_table = Table(
            [cards[:3], cards[3:]],
            colWidths=[doc.width / 3.0, doc.width / 3.0, doc.width / 3.0],
        )
        metrics_cards_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e8f7f3")),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#eef4ff")),
                    ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#fff5e9")),
                    ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#f3f7fb")),
                    ("BACKGROUND", (1, 1), (1, 1), colors.HexColor(risk["soft"])),
                    ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#edf5ff")),
                    ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#d6e0e8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(metrics_cards_table)
        story.append(Spacer(1, 9))

        detail_rows = [
            [
                Paragraph("<b>Share of Voice</b>", body_style),
                Paragraph(
                    self._dict_to_inline(
                        report.get("share_of_voice") or {}, suffix="%"
                    ),
                    body_style,
                ),
            ],
            [
                Paragraph("<b>Room Tone Profile</b>", body_style),
                Paragraph(
                    self._dict_to_inline(
                        report.get("room_tone_profile") or {}, suffix="%"
                    ),
                    body_style,
                ),
            ],
        ]
        details_table = Table(detail_rows, colWidths=[48 * mm, doc.width - 48 * mm])
        details_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4fb")),
                    ("BACKGROUND", (1, 0), (1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#d6e0ea")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(details_table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Final Summary", section_style))
        summary_text = self._safe_text(
            report.get("end_summary"),
            "No summary was generated.",
        )
        summary_box = Table(
            [[Paragraph(summary_text, body_style)]],
            colWidths=[doc.width],
        )
        summary_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fbff")),
                    ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor("#d3e0ec")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(summary_box)
        story.append(Spacer(1, 10))

        def add_bullets(title: str, items: Iterable[str]):
            item_list = [str(item).strip() for item in items if str(item).strip()]
            if not item_list:
                return
            story.append(Paragraph(title, section_style))
            story.append(
                ListFlowable(
                    [Paragraph(item, bullet_style) for item in item_list],
                    bulletType="bullet",
                    leftIndent=14,
                    bulletOffsetY=2,
                )
            )
            story.append(Spacer(1, 6))

        add_bullets("Key Observations", report.get("key_observations") or [])
        add_bullets(
            "Recommended Next Steps",
            report.get("recommended_next_steps")
            or report.get("suggested_next_steps")
            or [],
        )
        add_bullets("Action List", report.get("action_list") or [])

        story.append(Spacer(1, 6))
        note_box = Table(
            [
                [
                    Paragraph(
                        "No content recorded. Summary based on acoustic meeting metrics only.",
                        note_style,
                    )
                ]
            ],
            colWidths=[doc.width],
        )
        note_box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9ed")),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#ecdcb8")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(note_box)

        doc.build(story)
        return buffer.getvalue()
