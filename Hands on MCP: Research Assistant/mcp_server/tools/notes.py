"""
MCP Tools: save_note, read_notes
Persist research notes to disk and retrieve them — giving the agent
durable memory across sessions.
"""

from datetime import datetime
from pathlib import Path

# Resolve data/notes/ relative to this file's location (server/tools/notes.py)
NOTES_DIR = Path(__file__).parent.parent.parent / "data" / "notes"


def register(mcp) -> None:
    """Register save_note and read_notes as MCP tools."""

    @mcp.tool()
    def save_note(title: str, content: str) -> str:
        """
        Save a research note to disk as a Markdown file.

        Args:
            title: Short title for the note (used as the filename).
            content: The full note content in Markdown.

        Returns:
            Confirmation message with the saved file path.
        """
        NOTES_DIR.mkdir(parents=True, exist_ok=True)

        # Sanitise title → safe filename
        slug = title.lower().strip().replace(" ", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        filepath = NOTES_DIR / f"{slug}.md"

        timestamp = datetime.now().isoformat(timespec="seconds")
        note_text = f"# {title}\n\n*Saved: {timestamp}*\n\n{content}\n"
        filepath.write_text(note_text, encoding="utf-8")

        return f"✅ Note saved: {filepath.name}"

    @mcp.tool()
    def read_notes() -> str:
        """
        Read all saved research notes from disk.

        Returns:
            All notes concatenated as a single Markdown string,
            or a message if no notes exist yet.
        """
        if not NOTES_DIR.exists():
            return "No notes directory found. Save a note first."

        note_files = sorted(NOTES_DIR.glob("*.md"))
        if not note_files:
            return "No notes saved yet. Use save_note() to start building your research."

        sections = []
        for filepath in note_files:
            sections.append(
                f"=== {filepath.stem.replace('_', ' ').title()} ===\n\n"
                + filepath.read_text(encoding="utf-8")
            )

        return f"📚 Found {len(note_files)} note(s):\n\n" + "\n\n" + "─" * 60 + "\n\n".join(sections)
