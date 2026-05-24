"""
Tests for MCP Prompts: research_prompt, summarize_prompt.

Validates that:
- Prompts return the correct MCP PromptMessage structure.
- Dynamic arguments (topic, style) are interpolated correctly.
- Unknown styles fall back gracefully.
"""

import pytest
from fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mcp():
    return FastMCP("Test Server")


def _get_prompt_fn(mcp_instance: FastMCP, name: str):
    """Retrieve a registered prompt function by name."""
    for prompt in mcp_instance._prompt_manager._prompts.values():
        if prompt.name == name:
            return prompt.fn
    raise KeyError(f"Prompt '{name}' not found.")


# ── research_prompt ───────────────────────────────────────────────────────────

class TestResearchPrompt:
    def test_returns_list_of_prompt_messages(self, mcp):
        from mcp_server.prompts.research import register
        register(mcp)
        research_prompt = _get_prompt_fn(mcp, "research_prompt")

        result = research_prompt(topic="Agentic AI")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], PromptMessage)

    def test_message_role_is_user(self, mcp):
        from mcp_server.prompts.research import register
        register(mcp)
        research_prompt = _get_prompt_fn(mcp, "research_prompt")

        result = research_prompt(topic="LLMs")
        assert result[0].role == "user"

    def test_topic_is_interpolated_in_content(self, mcp):
        from mcp_server.prompts.research import register
        register(mcp)
        research_prompt = _get_prompt_fn(mcp, "research_prompt")

        result = research_prompt(topic="Quantum Computing")
        text = result[0].content.text

        assert "Quantum Computing" in text

    def test_content_is_text_content_type(self, mcp):
        from mcp_server.prompts.research import register
        register(mcp)
        research_prompt = _get_prompt_fn(mcp, "research_prompt")

        result = research_prompt(topic="RAG")
        assert isinstance(result[0].content, TextContent)
        assert result[0].content.type == "text"

    def test_prompt_mentions_key_tools(self, mcp):
        from mcp_server.prompts.research import register
        register(mcp)
        research_prompt = _get_prompt_fn(mcp, "research_prompt")

        result = research_prompt(topic="anything")
        text = result[0].content.text

        # The prompt should guide the agent to use both key tools
        assert "search_web" in text
        assert "save_note" in text


# ── summarize_prompt ──────────────────────────────────────────────────────────

class TestSummarizePrompt:
    def test_returns_list_of_prompt_messages(self, mcp):
        from mcp_server.prompts.summarize import register
        register(mcp)
        summarize_prompt = _get_prompt_fn(mcp, "summarize_prompt")

        result = summarize_prompt(style="bullet")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], PromptMessage)

    def test_bullet_style_mentions_bullet_points(self, mcp):
        from mcp_server.prompts.summarize import register
        register(mcp)
        summarize_prompt = _get_prompt_fn(mcp, "summarize_prompt")

        result = summarize_prompt(style="bullet")
        assert "bullet" in result[0].content.text.lower()

    def test_paragraph_style_mentions_paragraphs(self, mcp):
        from mcp_server.prompts.summarize import register
        register(mcp)
        summarize_prompt = _get_prompt_fn(mcp, "summarize_prompt")

        result = summarize_prompt(style="paragraph")
        assert "paragraph" in result[0].content.text.lower()

    def test_executive_style_mentions_executive(self, mcp):
        from mcp_server.prompts.summarize import register
        register(mcp)
        summarize_prompt = _get_prompt_fn(mcp, "summarize_prompt")

        result = summarize_prompt(style="executive")
        assert "executive" in result[0].content.text.lower()

    def test_unknown_style_falls_back_to_bullet(self, mcp):
        from mcp_server.prompts.summarize import register
        register(mcp)
        summarize_prompt = _get_prompt_fn(mcp, "summarize_prompt")

        # Unknown style should not raise; falls back to bullet
        result = summarize_prompt(style="unknown_style_xyz")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_default_style_is_bullet(self, mcp):
        from mcp_server.prompts.summarize import register
        register(mcp)
        summarize_prompt = _get_prompt_fn(mcp, "summarize_prompt")

        result = summarize_prompt()  # no style argument
        assert "bullet" in result[0].content.text.lower()

    def test_prompt_references_generate_report_tool(self, mcp):
        from mcp_server.prompts.summarize import register
        register(mcp)
        summarize_prompt = _get_prompt_fn(mcp, "summarize_prompt")

        result = summarize_prompt(style="bullet")
        assert "generate_report" in result[0].content.text
