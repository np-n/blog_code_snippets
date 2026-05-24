"""
MCP Prompt: summarize_prompt
A reusable prompt template that asks the agent to summarise
collected research notes in one of three styles:
  - bullet    → grouped bullet points (default)
  - paragraph → flowing prose
  - executive → brief executive summary with next steps

This demonstrates that MCP prompts are dynamic — the same
template produces different outputs based on the `style` argument.
"""

from mcp.types import PromptMessage, TextContent

STYLE_INSTRUCTIONS: dict[str, str] = {
    "bullet": (
        "Format the summary as **clear bullet points** grouped by theme.\n"
        "Each bullet should be one concise insight."
    ),
    "paragraph": (
        "Write the summary as **flowing paragraphs** with smooth transitions.\n"
        "Aim for a well-structured narrative, not a list."
    ),
    "executive": (
        "Write a brief **executive summary** (3–5 sentences max) covering:\n"
        "- The single most important finding\n"
        "- Key implications\n"
        "- Recommended next steps"
    ),
}


def register(mcp) -> None:
    """Register summarize_prompt as an MCP prompt template."""

    @mcp.prompt()
    def summarize_prompt(style: str = "bullet") -> list[PromptMessage]:
        """
        Summarisation prompt in bullet, paragraph, or executive style.

        Args:
            style: Output format — one of 'bullet', 'paragraph', 'executive'.
                   Defaults to 'bullet'.

        Returns:
            A list of PromptMessages instructing the agent to summarise
            all collected notes in the requested style.
        """
        format_instruction = STYLE_INSTRUCTIONS.get(
            style, STYLE_INSTRUCTIONS["bullet"]
        )

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "Summarise all the research notes you have collected.\n\n"
                        f"{format_instruction}\n\n"
                        "Your summary must include:\n"
                        "- **Core concepts** discovered during research\n"
                        "- **Key insights and patterns** across the notes\n"
                        "- **Notable sources or references** worth revisiting\n"
                        "- **Suggested areas** for further research\n\n"
                        "Use `read_notes()` first if you haven't already, "
                        "then call `generate_report()` to save the final output."
                    ),
                ),
            )
        ]
