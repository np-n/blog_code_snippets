"""
Research Assistant Agent
=========================
The main entry point for the LangChain ReAct agent.

Demo flow (mirrors the blog post step-by-step walkthrough):

  User: "Research Agentic AI for me"
    → reads  config://agent resource          ← 📦 Resource
    → reads  notes://all resource             ← 📦 Resource
    → fetches research_prompt(topic)          ← 📋 Prompt
    → fetches summarize_prompt("bullet")      ← 📋 Prompt
    → builds system prompt from the above
    → agent calls search_web("Agentic AI")    ← 🔧 Tool
    → agent calls save_note(...)              ← 🔧 Tool
    → agent calls read_notes()               ← 🔧 Tool
    → agent calls generate_report()          ← 🔧 Tool
    → prints final report ✅

Run:
    # Terminal 1 — start the MCP server first
    python -m server.server

    # Terminal 2 — run the agent
    python -m agent.agent "Agentic AI"
"""

import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from agent.llm import get_llm
from agent.mcp_client import get_mcp_client

load_dotenv()


def _extract_text(content) -> str:
    """
    Safely extract plain text from an MCP resource or prompt content object.
    Handles both string content and objects with a .text attribute.
    """
    if isinstance(content, str):
        return content
    if hasattr(content, "text"):
        return content.text
    return str(content)


def _build_system_prompt(config_text: str) -> str:
    """
    Build a SHORT role-only system prompt from the config://agent resource.

    Deliberately concise — llama-3.3-70b-versatile falls back to an XML
    tool-call format (<function=...>) when the system prompt is too long.
    Research instructions come from the MCP research_prompt template and
    are passed as the human message instead.
    """
    import json
    try:
        config = json.loads(config_text)
        name = config.get("agent_name", "ResearchBot")
        personality = config.get("personality", "You are a focused research assistant.")
    except Exception:
        name = "ResearchBot"
        personality = "You are a focused research assistant."

    return f"You are {name}, a personal AI research assistant. {personality}"


async def run_research_agent(topic: str) -> None:
    """
    Run the full research workflow for a given topic.

    Args:
        topic: The research topic (e.g. "Agentic AI").
    """
    print(f"\n🤖 Research Assistant starting up...")
    print(f"📌 Topic: {topic}\n")
    print("─" * 60)

    # As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient is not a
    # context manager. Use client.session(server_name) for a live session.
    client = get_mcp_client()

    async with client.session("research_assistant") as session:
        # ── 📦 Step 1: Read Resources ──────────────────────────────────────
        # config://agent  → personality injected into the system prompt
        # notes://all     → existing context appended to the human message
        print("📦 Reading config://agent resource...")
        config_result = await session.read_resource("config://agent")
        config_text = _extract_text(config_result.contents[0])

        print("📦 Reading notes://all resource...")
        notes_result = await session.read_resource("notes://all")
        existing_notes = _extract_text(notes_result.contents[0])

        # ── 📋 Step 2: Fetch research_prompt template from MCP Server ─────
        # The prompt IS the human message — keeps system prompt short and
        # prevents the XML tool-call format issue in llama-3.3-70b-versatile.
        print("📋 Fetching research_prompt template...")
        research_prompt_result = await session.get_prompt(
            "research_prompt", arguments={"topic": topic}
        )
        research_instruction = _extract_text(
            research_prompt_result.messages[0].content
        )

        # ── 🔧 Step 3: Get MCP Tools as LangChain tools ───────────────────
        print("🔧 Loading MCP tools as LangChain tools...")
        tools = await load_mcp_tools(session)
        print(f"   Loaded {len(tools)} tool(s): {[t.name for t in tools]}\n")

        # ── Build a SHORT system prompt (role only) ────────────────────────
        system_prompt = _build_system_prompt(config_text)

        # ── Compose human message: MCP prompt + existing notes context ─────
        notes_context = (
            f"\n\nExisting notes for context:\n{existing_notes}"
            if existing_notes and "📭" not in existing_notes and "No notes" not in existing_notes
            else ""
        )
        human_message = research_instruction + notes_context

        # ── 🤖 Step 4: Build and run the ReAct agent ──────────────────────
        llm = get_llm()
        agent = create_react_agent(llm, tools)

        print(f"🔬 Starting research on: '{topic}'")
        print("=" * 60)

        # Each ReAct iteration = 2 LangGraph nodes (LLM call + tool execution).
        # recursion_limit=50 → allows up to 24 tool calls before forcing a stop.
        # A typical research run uses ~12 (5 searches + 5 saves + read + report).
        run_config = {"recursion_limit": 50}

        result = await agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_message),
                ]
            },
            config=run_config,
        )

        # ── Print final response ───────────────────────────────────────────
        print("\n✅ Research complete!")
        print("=" * 60)
        final_message = result["messages"][-1].content
        print(final_message)


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Agentic AI"
    asyncio.run(run_research_agent(topic))
