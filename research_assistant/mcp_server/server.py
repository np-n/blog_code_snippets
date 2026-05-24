"""
Research Assistant MCP Server
==============================
The entry point for the FastMCP server.

This file wires together all three MCP primitives:
  🔧 Tools     — search_web, save_note, read_notes, generate_report
  📋 Prompts   — research_prompt, summarize_prompt
  📦 Resources — notes://all, config://agent

Run this server before starting the agent:
    python -m server.server

The server listens on http://localhost:8000/mcp (Streamable HTTP transport).
"""

import sys
from pathlib import Path


from fastmcp import FastMCP

from mcp_server.tools import register_tools
from mcp_server.prompts import register_prompts
from mcp_server.resources import register_resources

# ── Create the shared FastMCP instance ────────────────────────────────────────
mcp = FastMCP(
    name="Research Assistant MCP Server",
    instructions=(
        "A personal AI research assistant that can search the web, "
        "save notes, read saved notes, and generate research reports. "
        "Use the provided tools, prompts, and resources to conduct "
        "thorough and well-organised research on any topic."
    ),
)

# ── Register all MCP primitives ───────────────────────────────────────────────
register_tools(mcp)       # 🔧  search_web, save_note, read_notes, generate_report
register_prompts(mcp)     # 📋  research_prompt, summarize_prompt
register_resources(mcp)   # 📦  notes://all, config://agent

# ── Start the server ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting Research Assistant MCP Server...")
    print("📡 Transport : Streamable HTTP")
    print("🌐 Address   : http://localhost:8001/mcp")
    print("─" * 50)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
