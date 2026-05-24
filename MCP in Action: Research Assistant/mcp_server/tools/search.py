"""
MCP Tool: search_web
Uses DuckDuckGo (no API key needed) to search the web and return
formatted results the agent can reason over.
"""

try:
    from ddgs import DDGS          # ddgs >= 9 (official rename)
except ImportError:
    from duckduckgo_search import DDGS  # fallback for older installs


def _safe(text: str) -> str:
    """Return text with non-UTF-8 bytes replaced so the MCP response always decodes."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def register(mcp) -> None:
    """Register search_web as an MCP tool."""

    @mcp.tool()
    def search_web(query: str, max_results: int = 5) -> str:
        """
        Search the web for a query and return top results.

        Args:
            query: The search query string.
            max_results: Number of results to return (default 5).

        Returns:
            Formatted string of search results with title, snippet, and source URL.
        """
        results = []

        try:
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=max_results):
                    title = _safe(result.get("title", ""))
                    body = _safe(result.get("body", ""))
                    href = _safe(result.get("href", ""))
                    results.append(f"**{title}**\n{body}\nSource: {href}")
        except Exception as exc:
            return f"Search failed for '{query}': {exc}"

        if not results:
            return f"No results found for query: '{query}'"

        header = f"Search results for: '{query}'\n{'=' * 50}\n\n"
        return header + "\n\n---\n\n".join(results)
