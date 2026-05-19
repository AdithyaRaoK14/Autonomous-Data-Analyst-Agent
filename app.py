# app.py
import streamlit as st
from agent.graph import agent
from agent.state import AgentState
from agent.tools import _df_store
from agent.tracer import ExecutionTrace, render_trace_in_streamlit
from agent.logger import log
from agent.cache import cache
from agent.config import cfg
from langchain_core.messages import HumanMessage, ToolMessage
import pandas as pd
import os
import time
import re
import glob

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Autonomous Data Analyst Agent")
st.caption(
    "Upload any CSV · Ask questions in plain English · Agent writes and executes code to answer"
)


# ─── Session State Init ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],
        "agent_messages": [],
        "csv_path": None,
        "df_info": None,
        "plot_paths": [],
        "iteration_logs": [],
        "last_uploaded": None,
        "last_iteration_count": 0,
        "session_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Stable session ID for the logger
    if not st.session_state.session_id:
        import uuid
        st.session_state.session_id = uuid.uuid4().hex[:8]


init_session()


# ─── Helpers ───────────────────────────────────────────────────────────────────

TOOL_ICONS = {
    "load_and_inspect_data": "📂",
    "execute_python_code": "🐍",
    "get_column_statistics": "📈",
    "get_correlation_analysis": "🔗",
}

# Keywords that signal the response is expected to include a plot.
# If a cache hit has no surviving plot files for a visual query, the cache
# entry is silently bypassed so the agent regenerates fresh output.
_VISUAL_KEYWORDS = {
    "plot", "graph", "chart", "heatmap", "histogram",
    "visualize", "visualise", "visualization", "visualisation",
    "scatter", "bar", "pie", "boxplot", "distribution",
}


def _needs_visual(prompt: str) -> bool:
    """Return True if the prompt appears to request a visualisation."""
    lower = prompt.lower()
    return any(kw in lower for kw in _VISUAL_KEYWORDS)


def build_trace_from_result(result: dict, question: str, elapsed_s: float) -> ExecutionTrace:
    """
    Reconstruct an ExecutionTrace from the final agent state.
    Matches each AIMessage tool_call to its ToolMessage result by tool_call_id.
    """
    trace = ExecutionTrace(question=question)
    trace.total_elapsed_s = elapsed_s
    trace.iteration_count = result.get("iteration_count", 0)
    trace.error_count = result.get("error_count", 0)

    messages = result.get("messages", [])

    # Build id → ToolMessage lookup so we can match results to calls
    tool_results: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results[msg.tool_call_id] = msg

    for msg in messages:
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            continue
        for tc in msg.tool_calls:
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})
            # Show first arg value as the input preview (code, file_path, column_name…)
            input_preview = str(next(iter(args.values()), "")) if args else ""

            result_msg = tool_results.get(tc.get("id", ""))
            output_preview = ""
            success = True
            plot_generated = False
            duration_ms = 0.0

            if result_msg:
                output_preview = str(result_msg.content)
                success = not any(
                    tag in output_preview
                    for tag in ("Execution Error", "Security Error", "Error loading")
                )
                plot_generated = "[PLOT_SAVED:" in output_preview

            trace.add_step(
                tool_name=tool_name,
                tool_input_preview=input_preview,
                tool_output_preview=output_preview,
                duration_ms=duration_ms,
                success=success,
                plot_generated=plot_generated,
            )

    return trace


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Data")

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], key="file_uploader")

    if uploaded:
        save_path = f"sandbox/{uploaded.name}"
        os.makedirs("sandbox", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        df = pd.read_csv(save_path)
        st.session_state.csv_path = save_path
        st.session_state.df_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        if st.session_state.get("last_uploaded") != uploaded.name:
            st.session_state.messages = []
            st.session_state.agent_messages = []
            st.session_state.plot_paths = []
            st.session_state.last_uploaded = uploaded.name
            _df_store.clear()
            log.session_start(st.session_state.session_id, save_path)

        st.success(f"✅ {uploaded.name}")
        st.markdown(f"**{df.shape[0]:,} rows · {df.shape[1]} columns**")

        with st.expander("Preview (first 5 rows)", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        with st.expander("Column info", expanded=False):
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Nulls": df.isnull().sum().values,
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cache Control ─────────────────────────────────────────────────────────
    st.header("🗄️ Cache")
    force_refresh = st.checkbox(
        "🔄 Bypass cache",
        value=False,
        help=(
            "Tick this to skip any cached response and always run the agent "
            "fresh. Useful when you've updated the CSV or changed your code."
        ),
    )

    st.divider()

    # ── System Status Panel ───────────────────────────────────────────────────
    st.header("⚙️ System Status")

    cache_stats = cache.stats()
    status_rows = [
        ("Model",   f"`{cfg.model_name}`"),
        ("Temp",    f"`{cfg.temperature}`"),
        ("Cache",   "✅ enabled" if cfg.enable_cache else "❌ disabled"),
        ("Timeout", f"`{cfg.code_timeout_seconds}s`"),
        ("Max iter", f"`{cfg.max_iterations}`"),
        ("Cache entries",
         f"{cache_stats['live_entries']} live · {cache_stats['expired_entries']} expired"),
    ]
    for label, value in status_rows:
        st.markdown(f"**{label}:** {value}")

    st.divider()

    st.header("💡 Try asking")
    suggestions = [
        "Load the dataset and give me a complete overview.",
        "Is there any class imbalance? Show me a visualization.",
        "What are the top 5 features most correlated with the target variable? Show a heatmap.",
        "Are there significant outliers? Which rows should I investigate?",
        "Plot the distribution of all numeric columns.",
        "What preprocessing steps would you recommend before training an ML model?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sugg_{s[:25]}", use_container_width=True):
            st.session_state["pending_query"] = s

    st.divider()

    # ── Agent Debug + Recent Logs ─────────────────────────────────────────────
    with st.expander("🔧 Agent Debug", expanded=False):
        st.markdown(
            f"**Session ID:** `{st.session_state.session_id}`  \n"
            f"**Last iteration count:** `{st.session_state.get('last_iteration_count', 0)}`"
        )
        if st.session_state.get("iteration_logs"):
            for entry in st.session_state["iteration_logs"][-5:]:
                st.markdown(f"- {entry}")

    with st.expander("📋 Recent Logs", expanded=False):
        events = log.read_recent_events(15)
        if events:
            for event in reversed(events):
                ts = event.get("ts", "")[:19].replace("T", " ")
                evt_type = event.get("event", "?")
                detail = event.get("tool", event.get("context", ""))
                ok = event.get("success", True)
                icon = "✅" if ok else "❌"
                if evt_type == "security_block":
                    icon = "🚫"
                st.text(f"{icon} {ts}  [{evt_type}]  {detail}")
        else:
            st.caption("No log entries yet.")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.session_state.plot_paths = []
        st.rerun()


# ─── Chat History Display ──────────────────────────────────────────────────────

with st.container():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("plots"):
                cols = st.columns(min(len(msg["plots"]), 2))
                for i, plot_path in enumerate(msg["plots"]):
                    if os.path.exists(plot_path):
                        _ = cols[i % 2].image(
                            plot_path,
                            use_container_width=True
                        )
            # Replay trace for history messages if stored
            if msg.get("trace"):
                render_trace_in_streamlit(msg["trace"])


# ─── Input Handling ────────────────────────────────────────────────────────────

query_from_suggestion = st.session_state.pop("pending_query", None)
prompt = st.chat_input(
    "Ask anything about your data...") or query_from_suggestion

if prompt:
    if not st.session_state.csv_path:
        st.warning("⬆️ Upload a CSV file first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        # ── Cache lookup ─────────────────────────────────────────────────────
        # Skip cache if the user explicitly requested a fresh run.
        cached_result = None if force_refresh else cache.get(
            st.session_state.csv_path, prompt
        )

        # Also bypass the cache when the query expects a visualisation but
        # the cached entry has no surviving plot files on disk.
        if cached_result and _needs_visual(prompt):
            valid_cached_plots = [
                p for p in cached_result.get("plot_paths", [])
                if os.path.exists(p)
            ]
            if not valid_cached_plots:
                cached_result = None   # force live regeneration

        if cached_result:
            response_text = cached_result["response_text"]
            valid_plots = cached_result.get("plot_paths", [])

            status_placeholder.empty()
            st.info("⚡ Served from cache")
            response_placeholder.markdown(response_text)

            if valid_plots:
                plot_cols = st.columns(min(len(valid_plots), 2))
                for i, p in enumerate(valid_plots):
                    if os.path.exists(p):
                        _ = plot_cols[i % 2].image(
                            p,
                            use_container_width=True
                        )

            # Build a minimal cache-hit trace for display
            trace = ExecutionTrace(question=prompt)
            trace.cache_hit = True
            trace.iteration_count = cached_result.get("iteration_count", 0)
            render_trace_in_streamlit(trace)

            st.caption(
                f"⚡ Cache hit · {cached_result.get('iteration_count', 0)} steps "
                f"(original run)"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "plots": valid_plots,
                "trace": trace,
            })
            st.stop()

        # ── Live agent run ───────────────────────────────────────────────────
        status_placeholder.markdown("🤔 *Agent is thinking...*")


        visual_instruction = ""
        if _needs_visual(prompt):
            visual_instruction = (
                "\nVISUAL TASK: The user wants a visualization. "
                "You MUST call execute_python_code with matplotlib/seaborn code that creates "
                "the plot and saves it via plt.savefig(plot_path). "
                "Do NOT use get_correlation_analysis as a substitute for drawing. "
                "Do NOT describe a plot in text — produce it with execute_python_code.\n"
            )

        agent_prompt = (
            f"CSV file is at: {st.session_state.csv_path}\n"
            f"{visual_instruction}"
            f"IMPORTANT: Call tools to execute code. "
            f"Do not write code blocks in your text response.\n\n"
            f"User question: {prompt}"
        )
        new_message = HumanMessage(content=agent_prompt)

        initial_state = AgentState(
            messages=[new_message],
            csv_path=st.session_state.csv_path,
            df_shape=st.session_state.df_info["shape"] if st.session_state.df_info else None,
            df_columns=st.session_state.df_info["columns"] if st.session_state.df_info else None,
            df_dtypes=None,
            iteration_count=0,
            last_code_executed=None,
            last_tool_output=None,
            plot_paths=[],
            error_count=0,
            last_error=None,
        )

        start_time = time.time()
        session_id = st.session_state.session_id
        log.session_start(session_id, st.session_state.csv_path)

        try:
            # ── Stream the agent so we can show live tool-call status ────────
            result = None
            for state_snapshot in agent.stream(initial_state, stream_mode="values"):
                result = state_snapshot
                msgs = state_snapshot.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        names = [tc["name"] for tc in last_msg.tool_calls]
                        icons = " · ".join(
                            f"{TOOL_ICONS.get(n, '🔧')} `{n}`" for n in names
                        )
                        status_placeholder.markdown(f"*Calling: {icons}…*")

            elapsed = round(time.time() - start_time, 1)

            # ── Extract response text ────────────────────────────────────────
            final_message = result["messages"][-1]
            response_text = getattr(
                final_message, "content", str(final_message))
            response_text = re.sub(
                r'\[PLOT_SAVED:.*?\]', '', response_text).strip()

            status_placeholder.empty()
            response_placeholder.markdown(response_text)

            # ── Deduplicate and validate new plots ───────────────────────────
            # ── Validate plots from graph state only ────────────────────────
            state_plots = result.get("plot_paths", [])

            seen: set[str] = set()
            valid_plots: list[str] = []

            for p in state_plots:
                if p not in seen and os.path.exists(p):
                    seen.add(p)
                    valid_plots.append(p)

            if valid_plots:
                plot_cols = st.columns(min(len(valid_plots), 2))
                for i, plot_path in enumerate(valid_plots):
                    _ = plot_cols[i % 2].image(
                        plot_path,
                        use_container_width=True
                    )

            # ── Build and render execution trace ─────────────────────────────
            trace = build_trace_from_result(
                result, question=prompt, elapsed_s=elapsed)
            render_trace_in_streamlit(trace)

            # ── Persist to cache ─────────────────────────────────────────────
            cache.set(
                st.session_state.csv_path,
                prompt,
                response_text,
                valid_plots,
                iteration_count=result.get("iteration_count", 0),
            )

            # ── Update session state ─────────────────────────────────────────
            st.session_state.agent_messages = result["messages"]
            st.session_state.last_iteration_count = result.get(
                "iteration_count", 0)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "plots": valid_plots,
                "trace": trace,
            })

            # ── Log performance ──────────────────────────────────────────────
            log.performance(
                session_id=session_id,
                elapsed_s=elapsed,
                iterations=result.get("iteration_count", 0),
                plots_generated=len(valid_plots),
                error_count=result.get("error_count", 0),
            )

            st.caption(
                f"⏱️ {elapsed}s · {result.get('iteration_count', 0)} agent steps"
            )

        except Exception as e:
            status_placeholder.empty()
            log.error("app_invoke", exc=e, session_id=session_id)
            error_msg = f"Agent error: {str(e)}\n\nTry rephrasing your question."
            response_placeholder.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
