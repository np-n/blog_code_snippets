"""
MCP Tool: generate_report
The "wow moment" — combines all saved notes into a timestamped
Markdown report and writes it to disk.
"""

from datetime import datetime
from pathlib import Path

NOTES_DIR = Path(__file__).parent.parent.parent / "data" / "notes"
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def register(mcp) -> None:
    """Register generate_report as an MCP tool."""

    @mcp.tool()
    def generate_report() -> str:
        """
        Generate a consolidated Markdown report from all saved research notes.

        Reads every .md file in data/notes/, combines them into a single
        report document, saves it to data/report_<timestamp>.md, and
        returns the full report content.

        Returns:
            The complete report as a Markdown string, including the save path.
        """
        if not NOTES_DIR.exists():
            return "❌ No notes directory found. Save some notes first."

        note_files = sorted(NOTES_DIR.glob("*.md"))
        if not note_files:
            return "❌ No notes found. Use search_web() and save_note() to gather research first."

        # Collect all note content
        sections = []
        for filepath in note_files:
            sections.append(filepath.read_text(encoding="utf-8"))

        # Build the report
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = timestamp.strftime("%Y%m%d_%H%M%S")

        divider = "\n\n---\n\n"
        report_body = divider.join(sections)

        report = (
            f"# 📋 Research Report\n\n"
            f"**Generated:** {timestamp_str}  \n"
            f"**Notes compiled:** {len(note_files)}\n\n"
            f"---\n\n"
            f"{report_body}\n"
        )

        # Save report to disk
        report_path = DATA_DIR / f"report_{timestamp_file}.md"
        report_path.write_text(report, encoding="utf-8")

        return f"✅ Report saved to: {report_path.name}\n\n{report}"
