"""
Resources package — registers all MCP resources onto the FastMCP instance.
Each sub-module exposes a register(mcp) function.
"""

from mcp_server.resources import notes_resource, config_resource


def register_resources(mcp) -> None:
    """Register every resource with the shared FastMCP instance."""
    notes_resource.register(mcp)
    config_resource.register(mcp)
