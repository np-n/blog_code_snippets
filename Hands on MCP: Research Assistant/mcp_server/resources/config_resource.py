"""
MCP Resource: config://agent
Exposes the agent's personality and style configuration as a resource.

The agent client reads this resource before the ReAct loop starts,
injecting it into the system prompt so the agent behaves according
to the user-defined personality — no hardcoding required.
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "config.json"

DEFAULT_CONFIG = {
    "agent_name": "ResearchBot",
    "personality": "You are a focused, systematic research assistant. Always cite your sources.",
    "research_style": "systematic",
    "summary_style": "bullet",
    "max_search_results": 5,
}


def register(mcp) -> None:
    """Register the config://agent resource with the FastMCP instance."""

    @mcp.resource("config://agent")
    def agent_config() -> str:
        """
        Agent personality and style configuration.

        URI: config://agent

        Returns:
            The contents of data/config.json as a formatted JSON string.
            Falls back to sensible defaults if the file is missing.
        """
        if not CONFIG_PATH.exists():
            return json.dumps(DEFAULT_CONFIG, indent=2)

        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return json.dumps(config, indent=2)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {**DEFAULT_CONFIG, "_error": f"Invalid config.json: {exc}"},
                indent=2,
            )
