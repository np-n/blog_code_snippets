"""
Tools package — registers all MCP tools onto the FastMCP instance.
Each sub-module exposes a register(mcp) function.
"""

from mcp_server.tools import search, notes, report


def register_tools(mcp) -> None:
    """Register every tool with the shared FastMCP instance."""
    search.register(mcp)
    notes.register(mcp)
    report.register(mcp)
