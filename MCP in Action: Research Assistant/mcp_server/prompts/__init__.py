"""
Prompts package — registers all MCP prompt templates onto the FastMCP instance.
Each sub-module exposes a register(mcp) function.
"""

from mcp_server.prompts import research, summarize


def register_prompts(mcp) -> None:
    """Register every prompt template with the shared FastMCP instance."""
    research.register(mcp)
    summarize.register(mcp)
