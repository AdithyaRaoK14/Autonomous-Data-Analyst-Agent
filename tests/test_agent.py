# tests/test_agent.py

import time
import tempfile
import multiprocessing
from pathlib import Path

import pandas as pd

from agent.cache import QueryCache
from agent.logger import log
from agent.prompts import build_data_context
from agent.tracer import ExecutionTrace
from sandbox.executor import (
    validate_code,
    run_secure,
)

# macOS / multiprocessing safety
multiprocessing.set_start_method("spawn", force=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test dataframe
# ─────────────────────────────────────────────────────────────────────────────

TEST_DF = pd.DataFrame({
    "age": [21, 25, 30, 22],
    "salary": [50000, 65000, 80000, 52000],
    "target": [0, 1, 1, 0],
})


# ─────────────────────────────────────────────────────────────────────────────
# Cache Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_set_and_get(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)

    cache.set(
        csv_path="test.csv",
        question="What is the mean age?",
        response_text="Mean age is 24.5",
        plot_paths=[],
        iteration_count=2,
    )

    result = cache.get("test.csv", "What is the mean age?")

    assert result is not None
    assert result["response_text"] == "Mean age is 24.5"
    assert result["iteration_count"] == 2


def test_cache_expiration(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=1)

    cache.set(
        csv_path="test.csv",
        question="Q",
        response_text="A",
        plot_paths=[],
    )

    time.sleep(2)

    result = cache.get("test.csv", "Q")

    assert result is None


def test_cache_invalidate(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)

    cache.set(
        csv_path="test.csv",
        question="Delete me",
        response_text="A",
        plot_paths=[],
    )

    cache.invalidate("test.csv", "Delete me")

    result = cache.get("test.csv", "Delete me")

    assert result is None


def test_cache_stats(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)

    cache.set(
        csv_path="a.csv",
        question="Q1",
        response_text="A1",
        plot_paths=[],
    )

    stats = cache.stats()

    assert "total_entries" in stats
    assert stats["total_entries"] >= 1


def test_cache_key_changes_when_file_changes(tmp_path):
    """Cache key must differ after the CSV is modified.

    This proves that editing a dataset automatically invalidates all cached
    responses that referenced it, even when the file path is unchanged.
    """
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2")

    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)

    key1 = cache._key(str(csv_file), "What is the mean?")

    # Sleep 1 s so mtime actually advances on filesystems with 1-second
    # resolution (e.g. ext3, HFS+).
    time.sleep(1)
    csv_file.write_text("a,b\n3,4")

    key2 = cache._key(str(csv_file), "What is the mean?")

    assert key1 != key2, (
        "Cache key should change when the CSV is modified, "
        "but both keys are identical: %s" % key1
    )


def test_cache_miss_after_file_modified(tmp_path):
    """A cached result must not be returned after the source CSV changes."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2")

    cache_dir = tmp_path / "cache"
    cache = QueryCache(cache_dir=str(cache_dir), ttl=60)

    cache.set(
        csv_path=str(csv_file),
        question="What is the mean?",
        response_text="Mean is 1.5",
        plot_paths=[],
    )

    # Verify the entry is readable before modification.
    assert cache.get(str(csv_file), "What is the mean?") is not None

    time.sleep(1)
    csv_file.write_text("a,b\n9,10")   # file changed → new mtime

    result = cache.get(str(csv_file), "What is the mean?")

    assert result is None, (
        "Cache should return None after CSV is modified, got: %s" % result
    )


# ─────────────────────────────────────────────────────────────────────────────
# Security Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_blocked_import():
    safe, reason = validate_code("""
import os
print("bad")
""")

    assert safe is False
    assert "Blocked import" in reason


def test_blocked_builtin():
    safe, reason = validate_code("""
eval("2+2")
""")

    assert safe is False
    assert "Blocked function call" in reason


def test_blocked_attribute_access():
    safe, reason = validate_code("""
x.__class__
""")

    assert safe is False
    assert "Blocked attribute access" in reason


def test_blocked_subprocess_call():
    safe, reason = validate_code("""
import subprocess
subprocess.run(["ls"])
""")

    assert safe is False


def test_syntax_error_detection():
    safe, reason = validate_code("""
for
""")

    assert safe is False
    assert "Syntax error" in reason


# ─────────────────────────────────────────────────────────────────────────────
# Secure Execution Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_code_execution():
    result = run_secure("""
print(df['age'].mean())
""", TEST_DF)

    assert result.success is True
    assert "24.5" in result.output


def test_plot_generation():
    result = run_secure("""
plt.plot(df['age'])
print("plot done")
""", TEST_DF)

    assert result.success is True
    assert result.plot_path is not None


def test_timeout_enforcement():
    result = run_secure("""
while True:
    pass
""", TEST_DF, timeout=2)

    assert result.success is False
    assert result.timed_out is True


def test_dataframe_operations():
    result = run_secure("""
print(df.groupby('target')['salary'].mean())
""", TEST_DF)

    assert result.success is True
    assert "72500" in result.output


def test_execution_error_capture():
    result = run_secure("""
print(x)
""", TEST_DF)

    assert result.success is False
    assert "Execution Error" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Trace Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_execution_trace():
    trace = ExecutionTrace(question="Analyze salary")

    trace.add_step(
        tool_name="execute_python_code",
        tool_input_preview="print(df.mean())",
        tool_output_preview="salary 61750",
        success=True,
    )

    trace.add_step(
        tool_name="execute_python_code",
        tool_input_preview="retry",
        tool_output_preview="fixed",
        success=True,
    )

    summary = trace.summary()

    assert summary["total_steps"] == 2
    assert summary["retries"] == 1


def test_trace_plot_tracking():
    trace = ExecutionTrace(question="Plot salary")

    trace.add_step(
        tool_name="execute_python_code",
        tool_input_preview="plt.plot(df['salary'])",
        tool_output_preview="done",
        plot_generated=True,
    )

    summary = trace.summary()

    assert summary["plots_generated"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_data_context():
    context = build_data_context({
        "csv_path": "data.csv",
        "rows": 100,
        "cols": 3,
        "columns": ["a", "b", "c"],
        "numeric_cols": ["a", "b"],
        "categorical_cols": ["c"],
        "missing_summary": "None",
    })

    assert "100" in context
    assert "data.csv" in context
    assert "numeric columns" in context.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Logger Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_logger_runs_without_crashing():
    log.tool_call(
        "execute_python_code",
        code="print(df.head())",
        session_id="test-session"
    )

    log.tool_result(
        "execute_python_code",
        success=True,
        duration_ms=120,
        session_id="test-session",
        output_preview="ok",
    )

    assert True


def test_logger_recent_events():
    events = log.read_recent_events(5)

    assert isinstance(events, list)


# ─────────────────────────────────────────────────────────────────────────────
# Integration-style Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_analysis_flow():
    result = run_secure("""
mean_salary = df['salary'].mean()
print(f"Mean salary: {mean_salary}")
""", TEST_DF)

    assert result.success is True
    assert "61750" in result.output
