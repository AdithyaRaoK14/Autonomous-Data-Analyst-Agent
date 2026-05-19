# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.tools import (
    load_and_inspect_data,
    execute_python_code,
    get_column_statistics,
    get_correlation_analysis,
    _df_store,
)
from agent.prompts import ANALYST_SYSTEM_PROMPT, build_data_context
from typing import Literal

# Keywords that signal the user wants a visualisation
_VISUAL_KWS = {
    "plot", "chart", "graph", "heatmap", "histogram",
    "visualize", "visualise", "visualization", "visualisation",
    "scatter", "bar", "boxplot", "distribution", "show me",
}

# ─── Model Setup ───────────────────────────────────────────────────────────────

tools = [
    load_and_inspect_data,
    execute_python_code,
    get_column_statistics,
    get_correlation_analysis,
]

llm = ChatOllama(
    model="qwen2.5:7b",     # matches exactly what ollama list shows
    temperature=0,           # deterministic — critical for reliable tool calling
    num_predict=2048,
)

# tool_choice="any" forces the model to always call a tool instead of just typing
try:
    llm_with_tools = llm.bind_tools(tools, tool_choice="any")
except Exception:
    # fallback if this version of langchain-ollama doesn't support tool_choice
    llm_with_tools = llm.bind_tools(tools)


# ─── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    The thinking node. Decides what tool to call next based on conversation
    history and current state.
    """
    system_content = ANALYST_SYSTEM_PROMPT

    if state.get("df_columns") and "current" in _df_store:
        import pandas as pd
        df = _df_store["current"]
        numeric_cols = list(df.select_dtypes(include='number').columns)
        categorical_cols = list(df.select_dtypes(exclude='number').columns)
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].to_dict()

        context = build_data_context({
            "csv_path": state.get("csv_path", "unknown"),
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": state["df_columns"],
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "missing_summary": str(missing_cols) if missing_cols else "None",
        })
        system_content = ANALYST_SYSTEM_PROMPT + context

    messages = [SystemMessage(content=system_content)] + state["messages"]

    response = llm_with_tools.invoke(messages)

    new_state = {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }

    # Capture df metadata after first load
    if "current" in _df_store and not state.get("df_columns"):
        df = _df_store["current"]
        new_state["df_columns"] = list(df.columns)
        new_state["df_shape"] = df.shape

    return new_state


def track_plots_node(state: AgentState) -> dict:
    """
    Scans recent messages for [PLOT_SAVED:...] markers.
    Handles both string and list-type message content (LangChain v0.3+).
    """
    last_messages = state["messages"]
    new_plot_paths = []

    for msg in reversed(last_messages[-5:]):
        # content can be a string or a list of dicts in newer LangChain
        raw = getattr(msg, 'content', '')
        if isinstance(raw, list):
            content = ' '.join(
                part.get('text', '') if isinstance(part, dict) else str(part)
                for part in raw
            )
        else:
            content = str(raw)

        if "[PLOT_SAVED:" in content:
            import re
            matches = re.findall(r'\[PLOT_SAVED:(.*?)\]', content)
            new_plot_paths.extend(matches)

    if new_plot_paths:
        existing = state.get("plot_paths", [])
        return {"plot_paths": existing + new_plot_paths}

    return {}


def error_handler_node(state: AgentState) -> dict:
    """
    Detects repeated errors. After 3 failures injects a recovery message
    so the agent tries a different approach instead of looping.
    """
    last_messages = state["messages"]
    last_tool_output = ""

    for msg in reversed(last_messages[-3:]):
        content = str(getattr(msg, 'content', ''))
        if "Execution Error" in content:
            last_tool_output = content
            break

    error_count = state.get("error_count", 0)

    if "Execution Error" in last_tool_output:
        new_error_count = error_count + 1
        if new_error_count >= 3:
            recovery_msg = HumanMessage(
                content="You have encountered multiple errors. Please try a completely "
                        "different approach, or explain why you cannot answer the question."
            )
            return {
                "error_count": new_error_count,
                "messages": [recovery_msg],
                "last_error": last_tool_output,
            }
        return {"error_count": new_error_count}

    return {"error_count": 0}


def _user_wants_visual(state: AgentState) -> bool:
    """Return True if any HumanMessage in this turn contains a visual keyword."""
    for msg in state["messages"]:
        if getattr(msg, "type", "") == "human":
            text = str(getattr(msg, "content", "")).lower()
            if any(kw in text for kw in _VISUAL_KWS):
                return True
    return False


def _execute_was_called(state: AgentState) -> bool:
    """Return True if execute_python_code was already invoked this turn."""
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "execute_python_code":
                    return True
    return False


_ENFORCEMENT_MARKER = "VISUAL_ENFORCEMENT:"


def visual_enforcer_node(state: AgentState) -> dict:
    """
    Injects a correction message when the user requested a visualization
    but the agent never called execute_python_code.

    Runs once per turn (the marker prevents re-injection on the next loop).
    """
    if not _user_wants_visual(state):
        return {}

    if _execute_was_called(state) or state.get("plot_paths"):
        return {}

    # Prevent re-injection
    already_injected = any(
        _ENFORCEMENT_MARKER in str(getattr(msg, "content", ""))
        for msg in state["messages"]
    )
    if already_injected:
        return {}

    return {
        "messages": [
            HumanMessage(
                content=(
                    f"{_ENFORCEMENT_MARKER} The user requested a visualization. "
                    "You have NOT called execute_python_code yet. "
                    "You MUST call execute_python_code now with matplotlib/seaborn code "
                    "that creates the requested plot and saves it with plt.savefig(plot_path). "
                    "Do NOT call get_correlation_analysis instead — that only returns text numbers."
                )
            )
        ]
    }


MAX_ITERATIONS = 12


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route after agent node — go to tools or end the conversation turn."""

    last_message = state["messages"][-1]

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"

    if state.get("error_count", 0) >= 3:
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# ─── Build Graph ───────────────────────────────────────────────────────────────

def build_agent():
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("track_plots", track_plots_node)
    graph.add_node("visual_enforcer", visual_enforcer_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    graph.add_edge("tools", "track_plots")
    graph.add_edge("track_plots", "visual_enforcer")
    graph.add_edge("visual_enforcer", "error_handler")
    graph.add_edge("error_handler", "agent")

    return graph.compile()


agent = build_agent()
