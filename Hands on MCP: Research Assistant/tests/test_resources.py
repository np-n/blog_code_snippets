"""
Tests for MCP Resources: notes://all, config://agent.

Validates that:
- Resources return the correct string content.
- Custom URI schemes (notes://, config://) are registered.
- Edge cases (empty notes dir, missing config file) are handled gracefully.
"""

import json
from pathlib import Path

import pytest
from fastmcp import FastMCP


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mcp():
    return FastMCP("Test Server")


def _get_resource_fn(mcp_instance: FastMCP, uri: str):
    """Retrieve a registered resource function by URI."""
    for resource in mcp_instance._resource_manager._resources.values():
        if str(resource.uri) == uri:
            return resource.fn
    raise KeyError(
        f"Resource '{uri}' not found. "
        f"Registered: {[str(r.uri) for r in mcp_instance._resource_manager._resources.values()]}"
    )


# ── notes://all ───────────────────────────────────────────────────────────────

class TestNotesResource:
    @pytest.fixture(autouse=True)
    def patch_notes_dir(self, tmp_path, monkeypatch):
        import mcp_server.resources.notes_resource as mod
        monkeypatch.setattr(mod, "NOTES_DIR", tmp_path / "notes")
        self.notes_dir = tmp_path / "notes"

    def test_resource_registered_with_correct_uri(self, mcp):
        from mcp_server.resources.notes_resource import register
        register(mcp)
        # Should not raise
        _get_resource_fn(mcp, "notes://all")

    def test_returns_placeholder_when_no_notes(self, mcp):
        from mcp_server.resources.notes_resource import register
        register(mcp)
        all_notes = _get_resource_fn(mcp, "notes://all")

        result = all_notes()
        assert "No notes" in result or "📭" in result

    def test_returns_note_content(self, mcp):
        from mcp_server.resources.notes_resource import register
        register(mcp)
        all_notes = _get_resource_fn(mcp, "notes://all")

        self.notes_dir.mkdir(parents=True, exist_ok=True)
        (self.notes_dir / "agentic_ai.md").write_text("# Agentic AI\nKey finding here.")

        result = all_notes()
        assert "Agentic AI" in result
        assert "Key finding here." in result

    def test_returns_multiple_notes(self, mcp):
        from mcp_server.resources.notes_resource import register
        register(mcp)
        all_notes = _get_resource_fn(mcp, "notes://all")

        self.notes_dir.mkdir(parents=True, exist_ok=True)
        (self.notes_dir / "note_a.md").write_text("# Note A\nContent A.")
        (self.notes_dir / "note_b.md").write_text("# Note B\nContent B.")

        result = all_notes()
        assert "Content A." in result
        assert "Content B." in result
        assert "2 file(s)" in result

    def test_result_is_string(self, mcp):
        from mcp_server.resources.notes_resource import register
        register(mcp)
        all_notes = _get_resource_fn(mcp, "notes://all")

        assert isinstance(all_notes(), str)


# ── config://agent ────────────────────────────────────────────────────────────

class TestConfigResource:
    @pytest.fixture(autouse=True)
    def patch_config_path(self, tmp_path, monkeypatch):
        import mcp_server.resources.config_resource as mod
        self.config_path = tmp_path / "config.json"
        monkeypatch.setattr(mod, "CONFIG_PATH", self.config_path)

    def test_resource_registered_with_correct_uri(self, mcp):
        from mcp_server.resources.config_resource import register
        register(mcp)
        _get_resource_fn(mcp, "config://agent")

    def test_returns_default_when_config_missing(self, mcp):
        from mcp_server.resources.config_resource import register
        register(mcp)
        agent_config = _get_resource_fn(mcp, "config://agent")

        result = agent_config()
        parsed = json.loads(result)

        assert "agent_name" in parsed
        assert "personality" in parsed

    def test_reads_existing_config(self, mcp):
        from mcp_server.resources.config_resource import register
        register(mcp)
        agent_config = _get_resource_fn(mcp, "config://agent")

        custom = {
            "agent_name": "CustomBot",
            "personality": "A very curious bot.",
            "research_style": "exploratory",
            "summary_style": "paragraph",
            "max_search_results": 10,
        }
        self.config_path.write_text(json.dumps(custom))

        result = agent_config()
        parsed = json.loads(result)

        assert parsed["agent_name"] == "CustomBot"
        assert parsed["personality"] == "A very curious bot."
        assert parsed["max_search_results"] == 10

    def test_handles_invalid_json_gracefully(self, mcp):
        from mcp_server.resources.config_resource import register
        register(mcp)
        agent_config = _get_resource_fn(mcp, "config://agent")

        self.config_path.write_text("{ this is not valid json }")

        result = agent_config()
        parsed = json.loads(result)

        # Should return default config with an error field
        assert "_error" in parsed
        assert "agent_name" in parsed

    def test_result_is_valid_json_string(self, mcp):
        from mcp_server.resources.config_resource import register
        register(mcp)
        agent_config = _get_resource_fn(mcp, "config://agent")

        result = agent_config()
        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
