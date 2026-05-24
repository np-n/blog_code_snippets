"""
MCP Prompt: research_prompt
A reusable, parameterised prompt template that guides the agent
through a systematic 5-step research process for any topic.

This is the key MCP insight: prompts are server-side templates,
not hardcoded strings in your client code.
"""

from mcp.types import PromptMessage, TextContent


def register(mcp) -> None:
    """Register research_prompt as an MCP prompt template."""

    @mcp.prompt()
    def research_prompt(topic: str) -> list[PromptMessage]:
        """
        Systematic research workflow prompt for a given topic.

        Args:
            topic: The subject to research (e.g. "Agentic AI").

        Returns:
            A list of PromptMessages that instruct the agent to
            research the topic in a structured, thorough way.
        """
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        f"Research the topic: **{topic}**\n\n"
                        "Follow this systematic research process:\n\n"
                        "1. **Define the topic** — search for core definitions and key concepts.\n"
                        "2. **Find recent developments** — search for the latest news and breakthroughs.\n"
                        "3. **Explore applications** — search for real-world use cases and examples.\n"
                        "4. **Identify key players** — search for leading tools, frameworks, and organisations.\n"
                        "5. **Note future directions** — search for trends and what's coming next.\n\n"
                        "For each step:\n"
                        "- Use `search_web()` to gather information.\n"
                        "- Use `save_note()` to persist key findings immediately.\n"
                        "- Always include the source URL in your notes.\n\n"
                        "Be thorough but concise. Quality over quantity."
                    ),
                ),
            )
        ]
