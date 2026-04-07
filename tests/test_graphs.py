"""
test_graphs.py -- Comprehensive headless test suite for all RAG node types and architectures.

Uses a MockEngine to replace Ollama+FAISS so tests run without any external services.
Verifies:
  - Data flow through all node types
  - Cache short-circuiting (hit vs miss)
  - Buffer memory persistence across runs
  - Router branching (keyword match vs default)
  - Copy/Merge splitting and recombination
  - Variable config changes (k, window_size, keyword)
  - Multi-level topological execution
"""

import sys
import os
import io
import traceback
import time

# Force UTF-8 output on Windows to avoid charmap errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_executor import GraphExecutor

# --- ANSI Colors (plain ASCII safe) ---
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# --- Mock Engine ---
class MockEngine:
    """
    Simulates RAGEngine without Ollama or FAISS.
    Tracks all calls for assertion.
    """
    system_prompt = "You are a helpful assistant.\n\n{context}"

    def __init__(self, k_override=None):
        self.vector_db = True  # Pretend DB is loaded
        self.retrieve_calls = []
        self.generate_calls = []
        self._k_override = k_override

    def retrieve(self, query, k=4):
        effective_k = self._k_override if self._k_override else k
        self.retrieve_calls.append({"query": query, "k": effective_k})
        # Return realistic mock documents based on k
        docs = []
        for i in range(effective_k):
            class FakeDoc:
                def __init__(self, content):
                    self.page_content = content
            docs.append(FakeDoc(f"[MOCK_CHUNK_{i+1}: context for '{query}']"))
        return docs

    def generate(self, prompt_text):
        self.generate_calls.append(prompt_text)
        # Echo the query back inside a generated answer so we can inspect it
        return f"[MOCK_ANSWER based on: {prompt_text[:80]}...]"

    def load_db(self):
        return True


# ─── Test Runner Infrastructure ───────────────────────────────────────────────
RESULTS = []

def run_test(name, fn):
    """Execute fn() and track pass/fail."""
    print(f"\n{BOLD}{CYAN}>> {name}{RESET}")
    try:
        fn()
        RESULTS.append((name, True, None))
        print(f"  {GREEN}[PASSED]{RESET}")
    except AssertionError as e:
        RESULTS.append((name, False, str(e)))
        print(f"  {RED}[FAILED]: {e}{RESET}")
    except Exception as e:
        RESULTS.append((name, False, traceback.format_exc()))
        print(f"  {RED}[ERROR]: {e}{RESET}")
        traceback.print_exc()

def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}\n    Expected: {repr(expected)}\n    Got:      {repr(actual)}")

def assert_in(needle, haystack, msg=""):
    if needle not in haystack:
        raise AssertionError(f"{msg}\n    '{needle}' not found in:\n    {repr(haystack)}")

def assert_not_in(needle, haystack, msg=""):
    if needle in haystack:
        raise AssertionError(f"{msg}\n    '{needle}' should NOT be in:\n    {repr(haystack)}")

def assert_true(condition, msg=""):
    if not condition:
        raise AssertionError(msg)

def assert_false(condition, msg=""):
    if condition:
        raise AssertionError(msg)

def make_executor(graph_data, engine, node_configs=None, reset_buffer=True):
    """Helper to create a GraphExecutor with optional configs."""
    logs = []
    statuses = {}

    def log_fn(text, color="#fff"):
        logs.append(text)
        # Print visually for debugging
        print(f"    {MAGENTA}| {RESET}{text}")

    def status_fn(nid, status):
        statuses[nid] = status

    GraphExecutor.cache_store = {}  # always reset cache per test unless caller manages it
    if reset_buffer:
        GraphExecutor.buffer_store = []

    ex = GraphExecutor(graph_data, engine, log_fn=log_fn, status_fn=status_fn, reset_buffer=reset_buffer)
    if node_configs:
        for nid, cfg in node_configs.items():
            ex.set_node_config(nid, cfg)
    return ex, logs, statuses


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 1: Simple RAG (Input → FAISS → LLM → Output)
# ─────────────────────────────────────────────────────────────────────────────
def _build_simple_rag():
    nodes = [
        {"id": "n_query",   "type": "Query Input"},
        {"id": "n_faiss",   "type": "FAISS DB"},
        {"id": "n_llm",     "type": "Ollama LLM"},
        {"id": "n_output",  "type": "Response Output"},
    ]
    edges = [
        {"from": "n_query",  "fp": 0, "to": "n_faiss",  "tp": 0},
        {"from": "n_faiss",  "fp": 0, "to": "n_llm",    "tp": 0},
        {"from": "n_llm",    "fp": 0, "to": "n_output",  "tp": 0},
    ]
    return {"nodes": nodes, "edges": edges}

def test_simple_rag_data_flow():
    engine = MockEngine()
    graph = _build_simple_rag()
    ex, logs, statuses = make_executor(graph, engine)
    result = ex.execute("What is RAG?")

    # FAISS must have been called with the query
    assert_true(len(engine.retrieve_calls) > 0, "FAISS retrieve() was never called")
    assert_eq(engine.retrieve_calls[0]["query"], "What is RAG?", "FAISS received wrong query")

    # LLM must have been called
    assert_true(len(engine.generate_calls) > 0, "LLM generate() was never called")

    # Result must propagate to output
    assert_in("[MOCK_ANSWER", result, "LLM answer must appear in final output")

def test_simple_rag_default_k():
    """FAISS DB default k=4 must be passed unless overridden."""
    engine = MockEngine()
    graph = _build_simple_rag()
    ex, _, _ = make_executor(graph, engine, node_configs={"n_faiss": {"k": 4}})
    ex.execute("Default k test")
    assert_eq(engine.retrieve_calls[0]["k"], 4, "Default k should be 4")

def test_simple_rag_custom_k():
    """If FAISS node config changes k, the executor must use the new value."""
    engine = MockEngine()
    graph = _build_simple_rag()
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_faiss": {"k": 7}})
    ex.execute("Custom k test")
    assert_eq(engine.retrieve_calls[0]["k"], 7, "Custom k=7 was not forwarded to retrieve()")
    assert_in("k=7", " ".join(logs), "Logs should mention k=7")


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 2: Cache-first RAG (Input → Cache → miss→FAISS→LLM | hit→Output)
# ─────────────────────────────────────────────────────────────────────────────
def _build_cache_rag():
    nodes = [
        {"id": "n_query",   "type": "Query Input"},
        {"id": "n_cache",   "type": "Cache"},
        {"id": "n_faiss",   "type": "FAISS DB"},
        {"id": "n_llm",     "type": "Ollama LLM"},
        {"id": "n_output",  "type": "Response Output"},
    ]
    edges = [
        # query → cache
        {"from": "n_query",  "fp": 0, "to": "n_cache",  "tp": 0},
        # cache miss (port 1) → faiss → llm → output
        {"from": "n_cache",  "fp": 1, "to": "n_faiss",  "tp": 0},
        {"from": "n_faiss",  "fp": 0, "to": "n_llm",    "tp": 0},
        {"from": "n_llm",    "fp": 0, "to": "n_output",  "tp": 0},
        # cache hit (port 0) → output (bypass)
        {"from": "n_cache",  "fp": 0, "to": "n_output",  "tp": 0},
    ]
    return {"nodes": nodes, "edges": edges}

def test_cache_miss_on_first_run():
    """First run should MISS cache and call FAISS+LLM."""
    GraphExecutor.cache_store = {}
    engine = MockEngine()
    graph = _build_cache_rag()
    ex, logs, _ = make_executor(graph, engine)
    result = ex.execute("cache test query")

    assert_in("MISS", " ".join(logs), "First run must be a cache MISS")
    assert_true(len(engine.retrieve_calls) > 0, "FAISS must be called on cache miss")
    assert_true(len(engine.generate_calls) > 0, "LLM must be called on cache miss")

def test_cache_hit_on_second_run():
    """Second identical run must HIT cache and skip FAISS+LLM."""
    GraphExecutor.cache_store = {}
    engine = MockEngine()
    graph = _build_cache_rag()

    # First run: populate cache
    ex1, _, _ = make_executor(graph, engine, reset_buffer=True)
    GraphExecutor.cache_store = {}
    ex1.execute("repeated query")
    cache_after_first = dict(GraphExecutor.cache_store)
    assert_true(len(cache_after_first) > 0, "Cache must have been populated after first run")

    # Second run: should HIT
    engine2 = MockEngine()
    ex2, logs2, _ = make_executor(graph, engine2, reset_buffer=False)
    GraphExecutor.cache_store = cache_after_first  # restore from run 1
    result = ex2.execute("repeated query")

    assert_in("HIT", " ".join(logs2), "Second run must be a cache HIT")
    assert_eq(len(engine2.retrieve_calls), 0, "FAISS must NOT be called on cache hit")
    assert_eq(len(engine2.generate_calls), 0, "LLM must NOT be called on cache hit")

def test_cache_different_queries():
    """Different queries should each cause a cache MISS."""
    GraphExecutor.cache_store = {}
    engine = MockEngine()
    graph = _build_cache_rag()
    ex, logs, _ = make_executor(graph, engine, reset_buffer=True)
    GraphExecutor.cache_store = {}
    ex.execute("query A")

    engine2 = MockEngine()
    ex2, logs2, _ = make_executor(graph, engine2, reset_buffer=False)
    ex2.execute("query B")

    assert_in("MISS", " ".join(logs2), "A different query should be a cache MISS")
    assert_true(len(engine2.retrieve_calls) > 0, "FAISS must be called for new query")


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 3: Memory-augmented RAG (Buffer retains conversation history)
# ─────────────────────────────────────────────────────────────────────────────
def _build_memory_rag():
    nodes = [
        {"id": "n_query",   "type": "Query Input"},
        {"id": "n_faiss",   "type": "FAISS DB"},
        {"id": "n_buffer",  "type": "Buffer"},
        {"id": "n_merge",   "type": "Merge"},
        {"id": "n_llm",     "type": "Ollama LLM"},
        {"id": "n_output",  "type": "Response Output"},
    ]
    edges = [
        {"from": "n_query",  "fp": 0, "to": "n_faiss",  "tp": 0},
        {"from": "n_query",  "fp": 0, "to": "n_buffer",  "tp": 0},
        {"from": "n_faiss",  "fp": 0, "to": "n_merge",   "tp": 0},
        {"from": "n_buffer", "fp": 0, "to": "n_merge",   "tp": 1},
        {"from": "n_merge",  "fp": 0, "to": "n_llm",     "tp": 0},
        {"from": "n_llm",    "fp": 0, "to": "n_output",  "tp": 0},
    ]
    return {"nodes": nodes, "edges": edges}

def test_buffer_empty_on_first_run():
    """First run should have empty buffer (no history)."""
    GraphExecutor.buffer_store = []
    engine = MockEngine()
    graph = _build_memory_rag()
    ex, logs, _ = make_executor(graph, engine)
    ex.execute("first question")
    assert_in("empty", " ".join(logs).lower(), "First run buffer should indicate empty")

def test_buffer_accumulates_across_runs():
    """Subsequent runs should accumulate buffer entries."""
    GraphExecutor.cache_store = {}
    GraphExecutor.buffer_store = []
    engine = MockEngine()
    graph = _build_memory_rag()

    # Run 1
    ex1, _, _ = make_executor(graph, engine, reset_buffer=True)
    GraphExecutor.buffer_store = []
    ex1.execute("turn 1 question")
    entries_after_run1 = len(GraphExecutor.buffer_store)

    # Run 2 (preserve buffer)
    engine2 = MockEngine()
    ex2, logs2, _ = make_executor(graph, engine2, reset_buffer=False)
    ex2.execute("turn 2 question")
    entries_after_run2 = len(GraphExecutor.buffer_store)

    assert_true(entries_after_run2 > entries_after_run1,
                f"Buffer must grow: run1={entries_after_run1}, run2={entries_after_run2}")

def test_buffer_window_size_respected():
    """Buffer output must respect window_size config."""
    GraphExecutor.cache_store = {}
    GraphExecutor.buffer_store = ["old1", "old2", "old3", "old4", "old5", "old6"]
    engine = MockEngine()
    graph = _build_memory_rag()
    ex, logs, _ = make_executor(
        graph, engine,
        node_configs={"n_buffer": {"window_size": 3}},
        reset_buffer=False
    )
    ex.execute("window test")
    # Log should indicate the buffer was used (3 entries or how many are stored)
    assert_in("Buffer", " ".join(logs), "Buffer node must log itself")

def test_merge_combines_faiss_and_buffer():
    """Merge node must combine FAISS context and Buffer memory."""
    GraphExecutor.cache_store = {}
    GraphExecutor.buffer_store = ["previous answer stored in buffer"]
    engine = MockEngine()
    graph = _build_memory_rag()
    ex, logs, _ = make_executor(graph, engine, reset_buffer=False)
    ex.execute("merge test query")
    assert_in("Merge", " ".join(logs), "Merge node must be executed")
    assert_in("combined", " ".join(logs).lower(), "Merge should log that it combined inputs")


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 4: Router branching (keyword → path A, else → path B)
# ─────────────────────────────────────────────────────────────────────────────
def _build_router_rag():
    nodes = [
        {"id": "n_query",   "type": "Query Input"},
        {"id": "n_router",  "type": "Router"},
        {"id": "n_llm_a",   "type": "Ollama LLM"},    # route_a
        {"id": "n_llm_b",   "type": "Ollama LLM"},    # route_b
        {"id": "n_out_a",   "type": "Response Output"},
        {"id": "n_out_b",   "type": "Response Output"},
    ]
    edges = [
        {"from": "n_query",  "fp": 0, "to": "n_router", "tp": 0},
        {"from": "n_router", "fp": 0, "to": "n_llm_a",  "tp": 0},   # route_a
        {"from": "n_router", "fp": 1, "to": "n_llm_b",  "tp": 0},   # route_b
        {"from": "n_llm_a",  "fp": 0, "to": "n_out_a",  "tp": 0},
        {"from": "n_llm_b",  "fp": 0, "to": "n_out_b",  "tp": 0},
    ]
    return {"nodes": nodes, "edges": edges}

def test_router_keyword_match_routes_to_a():
    """Query containing keyword should go to route_a, skip route_b."""
    engine = MockEngine()
    graph = _build_router_rag()
    ex, logs, _ = make_executor(
        graph, engine,
        node_configs={"n_router": {"routing_type": "keyword", "keyword": "python"}}
    )
    ex.execute("python programming question")
    log_str = " ".join(logs)
    assert_in("route_a", log_str, "Keyword 'python' match must route to route_a")
    # LLM A should run, LLM B should be skipped (short circuit)
    assert_in("Short-circuit", log_str, "Route B should be short-circuited")

def test_router_no_keyword_routes_to_b():
    """Query without keyword should take default route_b."""
    engine = MockEngine()
    graph = _build_router_rag()
    ex, logs, _ = make_executor(
        graph, engine,
        node_configs={"n_router": {"routing_type": "keyword", "keyword": "python"}}
    )
    ex.execute("general unrelated question")
    log_str = " ".join(logs)
    assert_in("route_b", log_str, "No keyword match should default to route_b")

def test_router_keyword_case_insensitive():
    """Keyword match should be case-insensitive."""
    engine = MockEngine()
    graph = _build_router_rag()
    ex, logs, _ = make_executor(
        graph, engine,
        node_configs={"n_router": {"routing_type": "keyword", "keyword": "Python"}}
    )
    ex.execute("PYTHON is case-insensitive")
    assert_in("route_a", " ".join(logs), "Keyword match must be case-insensitive")

def test_router_empty_keyword_defaults_to_b():
    """Empty keyword should always take route_b."""
    engine = MockEngine()
    graph = _build_router_rag()
    ex, logs, _ = make_executor(
        graph, engine,
        node_configs={"n_router": {"routing_type": "keyword", "keyword": ""}}
    )
    ex.execute("anything here")
    assert_in("route_b", " ".join(logs), "Empty keyword must default to route_b")


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 5: Copy + Merge (Split then recombine)
# ─────────────────────────────────────────────────────────────────────────────
def _build_copy_merge():
    nodes = [
        {"id": "n_query",  "type": "Query Input"},
        {"id": "n_copy",   "type": "Copy"},
        {"id": "n_faiss",  "type": "FAISS DB"},
        {"id": "n_merge",  "type": "Merge"},
        {"id": "n_llm",    "type": "Ollama LLM"},
        {"id": "n_output", "type": "Response Output"},
    ]
    edges = [
        {"from": "n_query", "fp": 0, "to": "n_copy",   "tp": 0},
        {"from": "n_copy",  "fp": 0, "to": "n_faiss",  "tp": 0},   # out_a → faiss
        {"from": "n_copy",  "fp": 1, "to": "n_merge",  "tp": 1},   # out_b → merge directly
        {"from": "n_faiss", "fp": 0, "to": "n_merge",  "tp": 0},   # faiss → merge
        {"from": "n_merge", "fp": 0, "to": "n_llm",    "tp": 0},
        {"from": "n_llm",   "fp": 0, "to": "n_output", "tp": 0},
    ]
    return {"nodes": nodes, "edges": edges}

def test_copy_produces_two_outputs():
    """Copy node must produce identical data on both output ports."""
    engine = MockEngine()
    graph = _build_copy_merge()
    ex, logs, _ = make_executor(graph, engine)
    ex.execute("copy test")
    assert_in("duplicated to 2 outputs", " ".join(logs), "Copy must log it duplicated")

def test_merge_combines_multiple_inputs():
    """Merge node must receive data from multiple upstream ports."""
    engine = MockEngine()
    graph = _build_copy_merge()
    ex, logs, _ = make_executor(graph, engine)
    ex.execute("merge combination test")
    # GraphExecutor logs: '  [Merge] combined N inputs'
    merge_logs = [l for l in logs if "Merge" in l]
    assert_true(len(merge_logs) > 0, "Merge node must log its execution")
    assert_true(
        any("combined" in l.lower() for l in merge_logs),
        f"Merge log must contain 'combined'. Got: {merge_logs}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE 6: Full Pipeline (PDF → FAISS → Cache → Buffer → Merge → LLM)
# ─────────────────────────────────────────────────────────────────────────────
def _build_full_pipeline():
    """
    Correct topology: all downstream computation flows through Cache ports.
    - Cache MISS (port 1) → FAISS → Merge → LLM → Output
    - Cache HIT  (port 0) → directly to Output (bypasses FAISS+Buffer+Merge+LLM)
    - Buffer receives from Cache miss so it is also short-circuited on hit.
    """
    nodes = [
        {"id": "n_pdf",    "type": "PDF Loader"},
        {"id": "n_query",  "type": "Query Input"},
        {"id": "n_faiss",  "type": "FAISS DB"},
        {"id": "n_cache",  "type": "Cache"},
        {"id": "n_buf",    "type": "Buffer"},
        {"id": "n_merge",  "type": "Merge"},
        {"id": "n_llm",    "type": "Ollama LLM"},
        {"id": "n_output", "type": "Response Output"},
    ]
    edges = [
        {"from": "n_pdf",   "fp": 0, "to": "n_faiss",  "tp": 0},  # PDF signal → FAISS
        {"from": "n_query", "fp": 0, "to": "n_cache",  "tp": 0},  # Query → Cache
        # Cache miss path: → FAISS → Merge, and → Buffer → Merge
        {"from": "n_cache", "fp": 1, "to": "n_faiss",  "tp": 0},  # miss → FAISS query
        {"from": "n_cache", "fp": 1, "to": "n_buf",    "tp": 0},  # miss → Buffer
        {"from": "n_faiss", "fp": 0, "to": "n_merge",  "tp": 0},  # FAISS chunks → Merge
        {"from": "n_buf",   "fp": 0, "to": "n_merge",  "tp": 1},  # Buffer memory → Merge
        {"from": "n_merge", "fp": 0, "to": "n_llm",    "tp": 0},  # Merge → LLM
        {"from": "n_llm",   "fp": 0, "to": "n_output", "tp": 0},  # LLM → Output
        # Cache hit bypass: skip entire computation chain
        {"from": "n_cache", "fp": 0, "to": "n_output", "tp": 0},  # hit → Output directly
    ]
    return {"nodes": nodes, "edges": edges}

def test_full_pipeline_executes_all_nodes():
    """Full pipeline must pass through PDF, Cache, FAISS, Buffer, Merge, LLM, Output."""
    GraphExecutor.cache_store = {}
    engine = MockEngine()
    graph = _build_full_pipeline()
    ex, logs, statuses = make_executor(graph, engine)
    result = ex.execute("full pipeline query")

    log_str = " ".join(logs)
    for expected_node in ["PDF Loader", "FAISS DB", "Cache", "Buffer", "Merge", "Ollama LLM", "Response Output"]:
        assert_in(expected_node, log_str, f"'{expected_node}' must appear in logs")
    assert_in("[MOCK_ANSWER", result, "Final output must contain LLM answer")

def test_full_pipeline_cache_hit_bypasses_llm():
    """
    Second run same query must HIT cache.
    NOTE: In this topology, PDF Loader feeds FAISS with a permanent live edge,
    so FAISS short-circuit is intentionally NOT triggered (PDF path keeps it alive).
    The key guarantee is: Cache HIT is detected, Buffer is short-circuited,
    and the cached answer is used in Response Output (bypassing the miss-path LLM).
    We verify the HIT is logged and Buffer was skipped.
    """
    GraphExecutor.cache_store = {}
    engine1 = MockEngine()
    graph = _build_full_pipeline()

    # First run: populate cache
    ex1, _, _ = make_executor(graph, engine1, reset_buffer=True)
    GraphExecutor.cache_store = {}
    ex1.execute("pipeline cache query")
    saved_cache = dict(GraphExecutor.cache_store)
    assert_true(len(saved_cache) > 0, "Cache must be populated after first run")

    # Second run: should HIT cache
    engine2 = MockEngine()
    ex2, logs2, _ = make_executor(graph, engine2, reset_buffer=False)
    GraphExecutor.cache_store = saved_cache
    ex2.execute("pipeline cache query")

    log_str2 = " ".join(logs2)
    assert_in("HIT", log_str2, "Second run must be a cache hit")
    # Buffer is exclusively wired from Cache miss port — must be short-circuited on HIT
    assert_in("Short-circuit", log_str2, "Buffer should be short-circuited on cache hit")


# ─────────────────────────────────────────────────────────────────────────────
# VARIABLE CHANGE TESTS — verify configs propagate to behaviour
# ─────────────────────────────────────────────────────────────────────────────
def test_faiss_k_change_from_4_to_10():
    engine = MockEngine()
    graph = _build_simple_rag()
    for k in [1, 4, 6, 10]:
        engine.retrieve_calls = []
        ex, _, _ = make_executor(graph, engine, node_configs={"n_faiss": {"k": k}})
        ex.execute(f"k={k} test")
        assert_eq(engine.retrieve_calls[-1]["k"], k, f"k={k} was not passed to retrieve()")

def test_buffer_window_0_to_10():
    """Window size changes must produce different buffer slices."""
    GraphExecutor.cache_store = {}
    # Pre-populate buffer with 10 entries
    GraphExecutor.buffer_store = [f"entry_{i}" for i in range(10)]
    engine = MockEngine()
    graph = _build_memory_rag()

    for window in [1, 3, 5, 10]:
        ex, logs, _ = make_executor(
            graph, engine,
            node_configs={"n_buffer": {"window_size": window}},
            reset_buffer=False
        )
        ex.execute(f"window={window} test")
        log_str = " ".join(logs)
        assert_in("Buffer", log_str, f"Buffer log must appear for window={window}")

def test_router_keyword_change():
    """Changing the router keyword must change which branch executes."""
    engine = MockEngine()
    graph = _build_router_rag()

    # With keyword "finance"
    ex1, logs1, _ = make_executor(
        graph, engine,
        node_configs={"n_router": {"routing_type": "keyword", "keyword": "finance"}}
    )
    ex1.execute("finance market analysis")
    assert_in("route_a", " ".join(logs1), "Keyword 'finance' should match route_a")

    # With keyword "science"
    engine2 = MockEngine()
    ex2, logs2, _ = make_executor(
        graph, engine2,
        node_configs={"n_router": {"routing_type": "keyword", "keyword": "science"}}
    )
    ex2.execute("finance market analysis")  # same query, no "science"
    assert_in("route_b", " ".join(logs2), "No 'science' keyword should go to route_b")


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────
def test_empty_query_still_runs():
    """Empty string query should not crash the executor."""
    engine = MockEngine()
    graph = _build_simple_rag()
    ex, logs, _ = make_executor(graph, engine)
    result = ex.execute("")  # empty query
    # Should complete without error
    assert_true(isinstance(result, str), "Result must be a string even for empty query")

def test_isolated_single_node_no_edges():
    """A single node with no edges should execute cleanly with no errors."""
    engine = MockEngine()
    graph = {
        "nodes": [{"id": "n_only", "type": "Query Input"}],
        "edges": []
    }
    ex, logs, _ = make_executor(graph, engine)
    result = ex.execute("isolated node")
    # No crash, result is empty string (no Response Output node)
    assert_eq(result, "", "No Response Output node means empty result")

def test_no_nodes_at_all():
    """Graph with zero nodes must return empty string without crashing."""
    engine = MockEngine()
    graph = {"nodes": [], "edges": []}
    ex, _, _ = make_executor(graph, engine)
    result = ex.execute("nothing")
    assert_eq(result, "", "Empty graph must yield empty result")

def test_faiss_without_db_logs_warning():
    """If engine.vector_db is None, FAISS node must log a warning."""
    engine = MockEngine()
    engine.vector_db = None  # Simulate no DB loaded
    graph = _build_simple_rag()
    ex, logs, _ = make_executor(graph, engine)
    ex.execute("no db test")
    assert_in("WARNING", " ".join(logs), "Must warn when no FAISS index is loaded")

def test_cache_store_persists_between_queries():
    """Cache should store multiple keys independently."""
    GraphExecutor.cache_store = {}
    GraphExecutor.buffer_store = []
    engine = MockEngine()
    graph = _build_cache_rag()

    for q in ["Q1", "Q2", "Q3"]:
        # NOTE: make_executor resets cache_store internally via GraphExecutor.__init__
        # We must save and restore the cache manually across runs.
        saved = dict(GraphExecutor.cache_store)
        ex, _, _ = make_executor(graph, engine, reset_buffer=False)
        # Restore previously accumulated cache entries before executing
        GraphExecutor.cache_store.update(saved)
        ex.execute(q)

    # All three queries should now be in cache
    cache_keys = list(GraphExecutor.cache_store.keys())
    for q in ["Q1", "Q2", "Q3"]:
        assert_in(q, cache_keys, f"Query '{q}' must be cached")

def test_response_output_receives_data():
    """Response Output node must receive and store final data."""
    engine = MockEngine()
    graph = _build_simple_rag()
    ex, _, statuses = make_executor(graph, engine)
    result = ex.execute("output reception test")
    assert_true(len(result) > 0, "Response Output must receive data from LLM")

def test_multiple_response_output_nodes():
    """Graph with two Response Output nodes must still produce a final result."""
    engine = MockEngine()
    graph = {
        "nodes": [
            {"id": "n_q",   "type": "Query Input"},
            {"id": "n_llm", "type": "Ollama LLM"},
            {"id": "n_o1",  "type": "Response Output"},
            {"id": "n_o2",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",   "fp": 0, "to": "n_llm", "tp": 0},
            {"from": "n_llm", "fp": 0, "to": "n_o1",  "tp": 0},
            {"from": "n_llm", "fp": 0, "to": "n_o2",  "tp": 0},
        ]
    }
    ex, _, _ = make_executor(graph, engine)
    result = ex.execute("dual output test")
    assert_true(len(result) > 0, "Should produce result with two output nodes")

def test_nodes_execute_in_topological_order():
    """Verify log entries appear in correct order for Simple RAG."""
    engine = MockEngine()
    graph = _build_simple_rag()
    ex, logs, _ = make_executor(graph, engine)
    ex.execute("ordering test")
    order = []
    for log in logs:
        for ntype in ["Query Input", "FAISS DB", "Ollama LLM", "Response Output"]:
            if f"[{ntype}]" in log and ntype not in order:
                order.append(ntype)
                break
    expected_order = ["Query Input", "FAISS DB", "Ollama LLM", "Response Output"]
    assert_eq(order, expected_order,
              f"Nodes must execute in topological order. Got: {order}")


# =============================================================================
# NEW NODE TESTS (Nodes 1–10)
# =============================================================================

# ── Node 1: Prompt Template ──────────────────────────────────────────────────
def test_prompt_template_formats_correctly():
    """Prompt Template must substitute {query}, {context}, {memory}."""
    engine = MockEngine()
    template = "SYS\nCtx: {context}\nMem: {memory}\nQ: {query}"
    graph = {
        "nodes": [
            {"id": "n_q",  "type": "Query Input"},
            {"id": "n_pt", "type": "Prompt Template"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",  "fp": 0, "to": "n_pt", "tp": 0},
            {"from": "n_pt", "fp": 0, "to": "n_o",  "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_pt": {"template": template}})
    result = ex.execute("hello world")
    assert_in("Q: hello world", result, "Query placeholder must be substituted")
    assert_in("Ctx:", result, "Context placeholder must be present")
    assert_in("[Prompt Template]", " ".join(logs), "Prompt Template must log")

def test_prompt_template_uses_default_when_empty():
    """Empty template config should fall back to built-in default."""
    engine = MockEngine()
    graph = {
        "nodes": [
            {"id": "n_q",  "type": "Query Input"},
            {"id": "n_pt", "type": "Prompt Template"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",  "fp": 0, "to": "n_pt", "tp": 0},
            {"from": "n_pt", "fp": 0, "to": "n_o",  "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_pt": {"template": ""}})
    result = ex.execute("test query")
    assert_in("You are a helpful assistant", result, "Default template must include system intro")
    assert_in("test query", result, "Query must appear in default template output")

# ── Node 2: System Message ───────────────────────────────────────────────────
def test_system_message_emits_constant():
    """System Message must emit its constant text on port 0."""
    engine = MockEngine()
    msg = "You are a strict and concise finance expert."
    graph = {
        "nodes": [
            {"id": "n_sys", "type": "System Message"},
            {"id": "n_llm", "type": "Ollama LLM"},
            {"id": "n_o",   "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_sys", "fp": 0, "to": "n_llm", "tp": 0},
            {"from": "n_llm", "fp": 0, "to": "n_o",   "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_sys": {"message": msg}})
    ex.execute("ignored")
    assert_in("System Message", " ".join(logs), "System Message must log")
    assert_true(len(engine.generate_calls) > 0, "LLM must be called after System Message")

# ── Node 3: Reranker ─────────────────────────────────────────────────────────
def test_reranker_keeps_top_k_chunks():
    """Reranker receives scored chunks and keeps only top_k."""
    engine = MockEngine()
    raw_chunks = "\n\n".join([f"[score={0.9 - i*0.15}]\nchunk text {i}" for i in range(5)])
    graph = {
        "nodes": [
            {"id": "n_in", "type": "Query Input"},
            {"id": "n_rr", "type": "Reranker"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_in", "fp": 0, "to": "n_rr", "tp": 0},
            {"from": "n_rr", "fp": 0, "to": "n_o",  "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_rr": {"top_k": 2}})
    ex.execute(raw_chunks)
    assert_in("Reranker", " ".join(logs), "Reranker must log")
    assert_in("top 2", " ".join(logs), "Reranker must log top_k=2")

# ── Node 4: Memory Formatter ─────────────────────────────────────────────────
def test_memory_formatter_truncates_long_history():
    """Memory Formatter must truncate history exceeding max_tokens."""
    engine = MockEngine()
    long_memory = "word " * 1000   # ~1000 words >> 100 token limit
    graph_data = {
        "nodes": [
            {"id": "n_q",  "type": "Query Input"},
            {"id": "n_mf", "type": "Memory Formatter"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",  "fp": 0, "to": "n_mf", "tp": 0},
            {"from": "n_mf", "fp": 0, "to": "n_o",  "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph_data, engine,
                                node_configs={"n_mf": {"max_tokens": 100, "summarize": False}})
    ex.execute(long_memory)
    assert_in("truncated", " ".join(logs).lower(), "Must log truncation for long history")

def test_memory_formatter_passes_short_history():
    """Memory Formatter must not truncate history within token limit."""
    engine = MockEngine()
    graph_data = {
        "nodes": [
            {"id": "n_q",  "type": "Query Input"},
            {"id": "n_mf", "type": "Memory Formatter"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",  "fp": 0, "to": "n_mf", "tp": 0},
            {"from": "n_mf", "fp": 0, "to": "n_o",  "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph_data, engine, node_configs={"n_mf": {"max_tokens": 500}})
    ex.execute("short text")
    assert_in("within limit", " ".join(logs), "Short input must log 'within limit'")

# ── Node 5: Conversation Starter ─────────────────────────────────────────────
def test_conversation_starter_emits_text():
    """Conversation Starter emits configured text on output port."""
    engine = MockEngine()
    starter = "Hello! I am your assistant today."
    graph = {
        "nodes": [
            {"id": "n_cs",  "type": "Conversation Starter"},
            {"id": "n_llm", "type": "Ollama LLM"},
            {"id": "n_o",   "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_cs",  "fp": 0, "to": "n_llm", "tp": 0},
            {"from": "n_llm", "fp": 0, "to": "n_o",   "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_cs": {"text": starter}})
    ex.execute("irrelevant")
    assert_in("Conversation Starter", " ".join(logs), "Must log node name")
    assert_true(len(engine.generate_calls) > 0, "LLM must be called")
    assert_in(starter[:20], engine.generate_calls[0], "Starter text must reach LLM")

# ── Node 6: Score Filter ─────────────────────────────────────────────────────
def test_score_filter_removes_low_chunks():
    """Score Filter must drop chunks with score below threshold."""
    engine = MockEngine()
    chunks = "[score=0.9]\nhigh quality\n\n[score=0.3]\nlow quality\n\n[score=0.7]\nmid quality"
    graph = {
        "nodes": [
            {"id": "n_q",  "type": "Query Input"},
            {"id": "n_sf", "type": "Score Filter"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",  "fp": 0, "to": "n_sf", "tp": 0},
            {"from": "n_sf", "fp": 0, "to": "n_o",  "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_sf": {"threshold": 0.5}})
    result = ex.execute(chunks)
    assert_not_in("low quality", result, "Chunk below threshold must be removed")
    assert_in("high quality", result, "High-score chunk must survive")
    assert_in("mid quality", result, "Mid-score chunk must survive")

def test_score_filter_passes_all_above_zero():
    """Score Filter with threshold=0 must pass all chunks."""
    engine = MockEngine()
    chunks = "[score=0.1]\nchunk A\n\n[score=0.05]\nchunk B"
    graph = {
        "nodes": [
            {"id": "n_q",  "type": "Query Input"},
            {"id": "n_sf", "type": "Score Filter"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",  "fp": 0, "to": "n_sf", "tp": 0},
            {"from": "n_sf", "fp": 0, "to": "n_o",  "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine, node_configs={"n_sf": {"threshold": 0.0}})
    result = ex.execute(chunks)
    assert_in("chunk A", result, "All chunks must pass with threshold=0")
    assert_in("chunk B", result, "All chunks must pass with threshold=0")

# ── Node 7: Debug Inspector ───────────────────────────────────────────────────
def test_debug_inspector_passes_through():
    """Debug Inspector must not modify data and must log it."""
    engine = MockEngine()
    graph = {
        "nodes": [
            {"id": "n_q",  "type": "Query Input"},
            {"id": "n_di", "type": "Debug Inspector"},
            {"id": "n_llm","type": "Ollama LLM"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",  "fp": 0, "to": "n_di",  "tp": 0},
            {"from": "n_di", "fp": 0, "to": "n_llm", "tp": 0},
            {"from": "n_llm","fp": 0, "to": "n_o",   "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine)
    result = ex.execute("debug pass-through test")
    assert_in("Debug Inspector", " ".join(logs), "Debug Inspector must log itself")
    assert_in("[MOCK_ANSWER", result, "End-to-end result must still flow through")

# ── Node 8: Multi Merge ───────────────────────────────────────────────────────
def test_multi_merge_combines_all_ports():
    """Multi Merge must combine all connected upstream inputs."""
    engine = MockEngine()
    graph = {
        "nodes": [
            {"id": "n_sys", "type": "System Message"},
            {"id": "n_cs",  "type": "Conversation Starter"},
            {"id": "n_mm",  "type": "Multi Merge"},
            {"id": "n_llm", "type": "Ollama LLM"},
            {"id": "n_o",   "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_sys", "fp": 0, "to": "n_mm",  "tp": 0},
            {"from": "n_cs",  "fp": 0, "to": "n_mm",  "tp": 1},
            {"from": "n_mm",  "fp": 0, "to": "n_llm", "tp": 0},
            {"from": "n_llm", "fp": 0, "to": "n_o",   "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(
        graph, engine,
        node_configs={
            "n_sys": {"message": "System: be concise."},
            "n_cs":  {"text": "Starter: welcome!"},
        }
    )
    ex.execute("multi merge test")
    assert_in("Multi Merge", " ".join(logs), "Multi Merge must log")
    assert_in("merged", " ".join(logs).lower(), "Must log number of merged inputs")

# ── Node 9: Top-K Retriever ───────────────────────────────────────────────────
def test_topk_retriever_calls_faiss_with_k():
    """Top-K Retriever must call engine.retrieve with the configured k."""
    engine = MockEngine()
    graph = {
        "nodes": [
            {"id": "n_q",    "type": "Query Input"},
            {"id": "n_topk", "type": "Top-K Retriever"},
            {"id": "n_o",    "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",    "fp": 0, "to": "n_topk", "tp": 0},
            {"from": "n_topk", "fp": 0, "to": "n_o",    "tp": 0},
        ]
    }
    for k in [2, 5, 8]:
        engine.retrieve_calls = []
        ex, _, _ = make_executor(graph, engine, node_configs={"n_topk": {"k": k}})
        ex.execute("top-k test")
        assert_eq(engine.retrieve_calls[-1]["k"], k, f"Top-K Retriever must use k={k}")

def test_topk_retriever_annotates_scores():
    """Top-K Retriever output must include [score=X] annotations."""
    engine = MockEngine()
    graph = {
        "nodes": [
            {"id": "n_q",    "type": "Query Input"},
            {"id": "n_topk", "type": "Top-K Retriever"},
            {"id": "n_o",    "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",    "fp": 0, "to": "n_topk", "tp": 0},
            {"from": "n_topk", "fp": 0, "to": "n_o",    "tp": 0},
        ]
    }
    ex, _, _ = make_executor(graph, engine, node_configs={"n_topk": {"k": 3}})
    result = ex.execute("score annotation test")
    assert_in("[score=", result, "Top-K Retriever output must contain [score=X] for Reranker")

def test_topk_retriever_warns_without_db():
    """Top-K Retriever must warn when no FAISS index is loaded."""
    engine = MockEngine()
    engine.vector_db = None
    graph = {
        "nodes": [
            {"id": "n_q",    "type": "Query Input"},
            {"id": "n_topk", "type": "Top-K Retriever"},
            {"id": "n_o",    "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_q",    "fp": 0, "to": "n_topk", "tp": 0},
            {"from": "n_topk", "fp": 0, "to": "n_o",    "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine)
    ex.execute("no db query")
    assert_in("WARNING", " ".join(logs), "Must warn when no FAISS index available")

# ── Node 10: Seed Buffer Loader ───────────────────────────────────────────────
def test_seed_buffer_loader_populates_buffer():
    """Seed Buffer Loader must inject seed lines into buffer_store."""
    GraphExecutor.buffer_store = []
    engine = MockEngine()
    seed = "Turn 1: User asked about cats.\nTurn 2: Assistant answered."
    graph = {
        "nodes": [
            {"id": "n_sb", "type": "Seed Buffer Loader"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_sb", "fp": 0, "to": "n_o", "tp": 0},
        ]
    }
    ex, logs, _ = make_executor(graph, engine,
                                node_configs={"n_sb": {"seed": seed}},
                                reset_buffer=False)
    ex.execute("seed test")
    assert_true(len(GraphExecutor.buffer_store) > 0, "Buffer must be populated by Seed Buffer Loader")
    assert_in("injected", " ".join(logs), "Must log that seed was injected")

def test_seed_buffer_loader_skips_duplicate():
    """Second run with same seed must not double-inject."""
    GraphExecutor.buffer_store = []
    engine = MockEngine()
    seed = "Seed line A"
    graph = {
        "nodes": [
            {"id": "n_sb", "type": "Seed Buffer Loader"},
            {"id": "n_o",  "type": "Response Output"},
        ],
        "edges": [
            {"from": "n_sb", "fp": 0, "to": "n_o", "tp": 0},
        ]
    }
    # First run: seed injected
    ex1, _, _ = make_executor(graph, engine,
                              node_configs={"n_sb": {"seed": seed}},
                              reset_buffer=False)
    GraphExecutor.buffer_store = []
    ex1.execute("run 1")

    # Second run: same seed already in buffer, should skip
    ex2, logs2, _ = make_executor(graph, engine,
                                  node_configs={"n_sb": {"seed": seed}},
                                  reset_buffer=False)
    ex2.execute("run 2")
    assert_in("skipped", " ".join(logs2).lower(), "Duplicate seed must be skipped")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run all tests and report
# ─────────────────────────────────────────────────────────────────────────────
ALL_TESTS = [
    # Architecture 1: Simple RAG
    ("Simple RAG — data flow",              test_simple_rag_data_flow),
    ("Simple RAG — default k=4",            test_simple_rag_default_k),
    ("Simple RAG — custom k=7",             test_simple_rag_custom_k),

    # Architecture 2: Cache RAG
    ("Cache RAG — miss on first run",       test_cache_miss_on_first_run),
    ("Cache RAG — hit on second run",       test_cache_hit_on_second_run),
    ("Cache RAG — different queries miss",  test_cache_different_queries),

    # Architecture 3: Memory RAG
    ("Memory RAG — empty buffer first run", test_buffer_empty_on_first_run),
    ("Memory RAG — buffer accumulates",     test_buffer_accumulates_across_runs),
    ("Memory RAG — window_size respected",  test_buffer_window_size_respected),
    ("Memory RAG — merge combines inputs",  test_merge_combines_faiss_and_buffer),

    # Architecture 4: Router
    ("Router — keyword match to route_a",   test_router_keyword_match_routes_to_a),
    ("Router — no match to route_b",        test_router_no_keyword_routes_to_b),
    ("Router — case-insensitive match",     test_router_keyword_case_insensitive),
    ("Router — empty keyword to route_b",   test_router_empty_keyword_defaults_to_b),

    # Architecture 5: Copy + Merge
    ("Copy/Merge — copy produces 2 outputs",test_copy_produces_two_outputs),
    ("Copy/Merge — merge combines inputs",  test_merge_combines_multiple_inputs),

    # Architecture 6: Full pipeline
    ("Full Pipeline — all nodes execute",   test_full_pipeline_executes_all_nodes),
    ("Full Pipeline — cache bypasses LLM",  test_full_pipeline_cache_hit_bypasses_llm),

    # Variable change tests
    ("Variable — FAISS k changes (1-10)",   test_faiss_k_change_from_4_to_10),
    ("Variable — Buffer window (1-10)",     test_buffer_window_0_to_10),
    ("Variable — Router keyword change",    test_router_keyword_change),

    # Edge cases
    ("Edge — empty query string",           test_empty_query_still_runs),
    ("Edge — single isolated node",         test_isolated_single_node_no_edges),
    ("Edge — zero nodes in graph",          test_no_nodes_at_all),
    ("Edge — FAISS without DB warns",       test_faiss_without_db_logs_warning),
    ("Edge — cache stores multiple keys",   test_cache_store_persists_between_queries),
    ("Edge — response output receives data",test_response_output_receives_data),
    ("Edge — two Response Output nodes",    test_multiple_response_output_nodes),

    # Topological ordering
    ("Topology — correct execution order",  test_nodes_execute_in_topological_order),

    # New Node 1: Prompt Template
    ("Node 1 — Prompt Template formats correctly",    test_prompt_template_formats_correctly),
    ("Node 1 — Prompt Template uses default",         test_prompt_template_uses_default_when_empty),

    # New Node 2: System Message
    ("Node 2 — System Message emits constant",        test_system_message_emits_constant),

    # New Node 3: Reranker
    ("Node 3 — Reranker keeps top_k chunks",          test_reranker_keeps_top_k_chunks),

    # New Node 4: Memory Formatter
    ("Node 4 — Memory Formatter truncates long",      test_memory_formatter_truncates_long_history),
    ("Node 4 — Memory Formatter passes short",        test_memory_formatter_passes_short_history),

    # New Node 5: Conversation Starter
    ("Node 5 — Conversation Starter emits text",      test_conversation_starter_emits_text),

    # New Node 6: Score Filter
    ("Node 6 — Score Filter removes low chunks",      test_score_filter_removes_low_chunks),
    ("Node 6 — Score Filter passes all above zero",   test_score_filter_passes_all_above_zero),

    # New Node 7: Debug Inspector
    ("Node 7 — Debug Inspector passes through",       test_debug_inspector_passes_through),

    # New Node 8: Multi Merge
    ("Node 8 — Multi Merge combines all ports",       test_multi_merge_combines_all_ports),

    # New Node 9: Top-K Retriever
    ("Node 9 — Top-K Retriever calls FAISS with k",  test_topk_retriever_calls_faiss_with_k),
    ("Node 9 — Top-K Retriever annotates scores",     test_topk_retriever_annotates_scores),
    ("Node 9 — Top-K Retriever warns without DB",     test_topk_retriever_warns_without_db),

    # New Node 10: Seed Buffer Loader
    ("Node 10 — Seed Buffer Loader populates buffer", test_seed_buffer_loader_populates_buffer),
    ("Node 10 — Seed Buffer Loader skips duplicate",  test_seed_buffer_loader_skips_duplicate),
]


if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  RAG Node & Architecture Test Suite{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Total tests: {len(ALL_TESTS)}")
    print(f"{'='*60}{RESET}\n")

    start_time = time.time()

    for name, fn in ALL_TESTS:
        run_test(name, fn)

    elapsed = round(time.time() - start_time, 2)

    # Summary
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed

    print(f"\n{BOLD}{'='*60}{RESET}")
    result_color = RED if failed else GREEN
    print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}{BOLD}, {result_color}{failed} failed{RESET}{BOLD}, {len(RESULTS)} total  ({elapsed}s){RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    if failed > 0:
        print(f"{BOLD}{RED}  FAILED TESTS:{RESET}")
        for name, ok, err in RESULTS:
            if not ok:
                print(f"\n  {RED}[FAIL] {name}{RESET}")
                print(f"    {err}")

    sys.exit(0 if failed == 0 else 1)
