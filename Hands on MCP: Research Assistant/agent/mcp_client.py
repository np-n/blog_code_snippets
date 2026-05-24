"""
MCP Client — LangChain ↔ MCP Bridge
======================================
Connects the LangChain agent to the running MCP server via Streamable HTTP transport.

As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient is NOT a context manager.
Use client.session(server_name) to get an async session context instead:

    from langchain_mcp_adapters.tools import load_mcp_tools

    client = get_mcp_client()
    async with client.session("research_assistant") as session:
        tools  = await load_mcp_tools(session)
        config = await session.read_resource("config://agent")
"""

import os

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# Default: matches the server started by `python -m server.server`
# FastMCP exposes the Streamable HTTP endpoint at /mcp
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")


def get_mcp_client() -> MultiServerMCPClient:
    """
    Create a MultiServerMCPClient pointed at the Research Assistant MCP server.

    The client connects via Streamable HTTP transport, so the MCP server
    must be running before this client is used.

    Returns:
        A MultiServerMCPClient instance. Use client.session(server_name)
        as an async context manager to obtain a live MCP session.
    """
    return MultiServerMCPClient(
        {
            "research_assistant": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    )
