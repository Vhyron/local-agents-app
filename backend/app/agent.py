from dataclasses import dataclass, field
import ollama
from duckduckgo_search import DDGS
import trafilatura


# ─── Tools ────────────────────────────────────────────────────────────

def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return a formatted list of results."""
    try:
        results = DDGS().text(query, max_results=num_results)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r['title']}\n    URL: {r['href']}\n    {r['body']}"
        )
    return "\n\n".join(formatted)


def read_url(url: str) -> str:
    """Fetch a URL and return its main text content, truncated."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Could not fetch {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"Could not extract text from {url}"
        return text[:4000]
    except Exception as e:
        return f"Error reading {url}: {e}"


# Tool registry: name → (function, JSON schema for Ollama)
TOOL_REGISTRY = {
    "web_search": (
        web_search,
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web and return a list of results with titles, URLs, and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "num_results": {"type": "integer", "description": "Number of results, default 5"},
                    },
                    "required": ["query"],
                },
            },
        },
    ),
    "read_url": (
        read_url,
        {
            "type": "function",
            "function": {
                "name": "read_url",
                "description": "Fetch a URL and return its main text content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to read"},
                    },
                    "required": ["url"],
                },
            },
        },
    ),
}


def available_tool_names() -> list[str]:
    """Return the list of all registered tool names. Used by the API."""
    return list(TOOL_REGISTRY.keys())


# ─── Defaults (used when no agent config is provided) ─────────────────

DEFAULT_SYSTEM_PROMPT = """You are a concise research assistant. Your job is to find verified information and report it tightly.

RESEARCH PROCESS:
1. Use web_search to find relevant sources.
2. Use read_url on 2-3 of the most credible results.
3. Cross-reference before stating anything as fact.
4. Prefer primary sources (official sites, peer-reviewed work, government data) over blogs and aggregators.

OUTPUT RULES:
- Be direct. No preamble, no "Great question!", no restating the question.
- Lead with the answer in 1-2 sentences.
- Follow with 3-5 bullet points of supporting facts only if they add value.
- End with sources as a numbered list of URLs.
- If you couldn't verify something, say "Unverified:" and flag it. Don't pad.
- No hedging filler ("it's important to note that...", "as we can see..."). Cut it.

OUTPUT FORMAT:

**Answer:** [1-2 sentence direct answer]

**Key points:**
- [fact]
- [fact]

**Sources:**
1. [URL]
2. [URL]

If sources disagree, add a "**Conflicts:**" section before Sources and state the disagreement plainly.

MODIFIERS:
- If the user ends their question with [BRIEF], give only the Answer + Sources sections.
- If they end with [DEEP DIVE], ignore the concise rules and write a thorough report.
- Otherwise, follow the default format above."""


@dataclass
class AgentConfig:
    """Runtime config for an agent. Built from a DB Agent row or used as default."""
    model: str = "qwen2.5:7b"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.3
    enabled_tools: list[str] = field(default_factory=lambda: ["web_search", "read_url"])
    max_steps: int = 8


# ─── Agent loop ───────────────────────────────────────────────────────

def run_agent(user_question: str, config: AgentConfig | None = None) -> str | None:
    """Run the agent loop. Returns the final answer text, or None if max steps hit."""
    if config is None:
        config = AgentConfig()

    # Build the tool list for this run from the agent's enabled tools
    tools_for_ollama = []
    available_tools = {}
    for tool_name in config.enabled_tools:
        if tool_name not in TOOL_REGISTRY:
            print(f"Warning: unknown tool '{tool_name}', skipping")
            continue
        func, schema = TOOL_REGISTRY[tool_name]
        available_tools[tool_name] = func
        tools_for_ollama.append(schema)

    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_question},
    ]

    for step in range(config.max_steps):
        response = ollama.chat(
            model=config.model,
            messages=messages,
            tools=tools_for_ollama if tools_for_ollama else None,
            options={"temperature": config.temperature},
        )
        msg = response["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                print(f"\n[Step {step + 1}] Calling {name}({args})")

                if name in available_tools:
                    try:
                        result = available_tools[name](**args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = f"Tool '{name}' is not enabled for this agent"

                messages.append({"role": "tool", "content": str(result)})
        else:
            print("\n=== ANSWER ===\n")
            print(msg["content"])
            return msg["content"]

    print("\nHit max steps without a final answer.")
    return None


# ─── Terminal entry point (kept for standalone testing) ───────────────

if __name__ == "__main__":
    print("Research agent ready. Type your question (or 'quit' to exit).")
    print("Modifiers: end with [BRIEF] for short answers, [DEEP DIVE] for thorough.\n")

    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        run_agent(question)
        print()