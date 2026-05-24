"""
MCP Resource: notes://all
Exposes all saved research notes as a readable resource.

Resources in MCP are like read-only data endpoints — the client
(or agent) can pull them to inject context before acting.
Unlike tools, resources are NOT called by the agent during its loop;
they are read by the client before the agent starts.
"""

from pathlib import Path

NOTES_DIR = Path(__file__).parent.parent.parent / "data" / "notes"


def register(mcp) -> None:
    """Register the notes://all resource with the FastMCP instance."""

    @mcp.resource("notes://all")
    def all_notes() -> str:
        """
        All saved research notes as a single readable resource.

        URI: notes://all

        Returns:
            Every .md file in data/notes/ concatenated into one
            Markdown string, or a placeholder if no notes exist yet.
        """
        if not NOTES_DIR.exists():
            return "📭 No notes directory found. Notes will appear here once saved."

        note_files = sorted(NOTES_DIR.glob("*.md"))
        if not note_files:
            return "📭 No notes saved yet. The agent will populate this resource during research."

        sections = []
        for filepath in note_files:
            title = filepath.stem.replace("_", " ").title()
            sections.append(f"### {title}\n\n{filepath.read_text(encoding='utf-8')}")

        divider = "\n\n---\n\n"
        header = f"# Research Notes ({len(note_files)} file(s))\n\n"
        return header + divider.join(sections)
