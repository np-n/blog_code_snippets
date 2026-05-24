"""
Tests for MCP Tools: search_web, save_note, read_notes, generate_report.

Strategy:
- Create a fresh FastMCP instance per test and call register(mcp).
- Invoke the registered functions directly (no HTTP involved).
- Mock DuckDuckGo so tests run offline.
- Use tmp_path (pytest built-in) for file-system isolation.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import FastMCP


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mcp():
    """A fresh FastMCP instance for each test."""
    return FastMCP("Test Server")


def _get_tool(mcp_instance: FastMCP, name: str):
    """Retrieve a registered tool function by name from the FastMCP instance."""
    for tool in mcp_instance._tool_manager._tools.values():
        if tool.name == name:
            return tool.fn
    raise KeyError(f"Tool '{name}' not found. Registered: {list(mcp_instance._tool_manager._tools)}")


# ── search_web ────────────────────────────────────────────────────────────────

class TestSearchWeb:
    def test_returns_formatted_results(self, mcp):
        from mcp_server.tools.search import register
        register(mcp)
        search_web = _get_tool(mcp, "search_web")

        fake_results = [
            {"title": "AI Overview", "body": "AI is everywhere.", "href": "https://example.com"},
            {"title": "Deep Learning", "body": "Neural networks.", "href": "https://dl.com"},
        ]

        with patch("server.tools.search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results
            result = search_web("artificial intelligence", max_results=2)

        assert "AI Overview" in result
        assert "Deep Learning" in result
        assert "https://example.com" in result
        assert "artificial intelligence" in result

    def test_returns_no_results_message(self, mcp):
        from mcp_server.tools.search import register
        register(mcp)
        search_web = _get_tool(mcp, "search_web")

        with patch("server.tools.search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = []
            result = search_web("xyzzy404notfound")

        assert "No results found" in result

    def test_default_max_results_is_five(self, mcp):
        from mcp_server.tools.search import register
        register(mcp)
        search_web = _get_tool(mcp, "search_web")

        call_args = {}

        def capture(**kwargs):
            call_args.update(kwargs)
            return []

        with patch("server.tools.search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.side_effect = (
                lambda q, max_results: call_args.update({"max_results": max_results}) or []
            )
            search_web("test query")

        assert call_args.get("max_results") == 5


# ── save_note & read_notes ────────────────────────────────────────────────────

class TestNotesTools:
    @pytest.fixture(autouse=True)
    def patch_notes_dir(self, tmp_path, monkeypatch):
        """Redirect NOTES_DIR to a temporary directory for every test."""
        import mcp_server.tools.notes as notes_module
        monkeypatch.setattr(notes_module, "NOTES_DIR", tmp_path / "notes")

    def test_save_note_creates_markdown_file(self, mcp, tmp_path):
        from mcp_server.tools.notes import register
        register(mcp)
        save_note = _get_tool(mcp, "save_note")

        result = save_note(title="Agentic AI", content="AI agents are autonomous systems.")

        assert "✅" in result
        note_files = list((tmp_path / "notes").glob("*.md"))
        assert len(note_files) == 1
        content = note_files[0].read_text()
        assert "Agentic AI" in content
        assert "AI agents are autonomous systems." in content

    def test_save_note_sanitises_title(self, mcp, tmp_path):
        from mcp_server.tools.notes import register
        register(mcp)
        save_note = _get_tool(mcp, "save_note")

        save_note(title="Hello World! 2024", content="test")

        note_files = list((tmp_path / "notes").glob("*.md"))
        assert len(note_files) == 1
        # Filename should be safe (no spaces or special chars)
        assert " " not in note_files[0].name
        assert "!" not in note_files[0].name

    def test_read_notes_returns_all_notes(self, mcp, tmp_path):
        from mcp_server.tools.notes import register
        register(mcp)
        save_note = _get_tool(mcp, "save_note")
        read_notes = _get_tool(mcp, "read_notes")

        save_note(title="Note One", content="First note content.")
        save_note(title="Note Two", content="Second note content.")

        result = read_notes()

        assert "First note content." in result
        assert "Second note content." in result
        assert "2 note(s)" in result

    def test_read_notes_empty(self, mcp, tmp_path):
        from mcp_server.tools.notes import register
        register(mcp)
        read_notes = _get_tool(mcp, "read_notes")

        result = read_notes()
        assert "No notes" in result


# ── generate_report ───────────────────────────────────────────────────────────

class TestGenerateReport:
    @pytest.fixture(autouse=True)
    def patch_dirs(self, tmp_path, monkeypatch):
        import mcp_server.tools.report as report_module
        monkeypatch.setattr(report_module, "NOTES_DIR", tmp_path / "notes")
        monkeypatch.setattr(report_module, "DATA_DIR", tmp_path)

    def _write_note(self, notes_dir: Path, name: str, content: str) -> None:
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / f"{name}.md").write_text(content)

    def test_generates_report_file(self, mcp, tmp_path):
        from mcp_server.tools.report import register
        register(mcp)
        generate_report = _get_tool(mcp, "generate_report")

        self._write_note(tmp_path / "notes", "note1", "# Note One\nContent here.")

        result = generate_report()

        assert "✅ Report saved" in result
        report_files = list(tmp_path.glob("report_*.md"))
        assert len(report_files) == 1

    def test_report_contains_note_content(self, mcp, tmp_path):
        from mcp_server.tools.report import register
        register(mcp)
        generate_report = _get_tool(mcp, "generate_report")

        self._write_note(tmp_path / "notes", "ai_note", "# AI\nAI is transformative.")
        result = generate_report()

        assert "AI is transformative." in result

    def test_report_fails_gracefully_with_no_notes(self, mcp, tmp_path):
        from mcp_server.tools.report import register
        register(mcp)
        generate_report = _get_tool(mcp, "generate_report")

        result = generate_report()
        assert "❌" in result
