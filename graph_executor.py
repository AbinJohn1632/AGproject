"""
graph_executor.py — Real graph-tracing execution engine.
Reads the visual node graph, walks it topologically, and routes data
through each node's actual logic (Cache, Buffer, Router, etc.).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


class GraphExecutor:
    """
    Executes a node graph by tracing actual visual connections.
    
    Shared state (persists across Quick Runs within a session):
        - cache_store: dict mapping query -> answer
        - buffer_store: list of conversation turns
    
    Full Run resets buffer_store; Quick Run preserves both.
    """

    # Class-level shared state (persists across runs within session)
    cache_store = {}
    buffer_store = []

    def __init__(self, graph_data, engine, log_fn=None, status_fn=None, reset_buffer=False):
        """
        graph_data: output of NodeScene.serialize() — {nodes: [...], edges: [...]}
        engine: RAGEngine instance with .retrieve(q,k) and .generate(prompt)
        log_fn: callable(text, color) for terminal logging
        status_fn: callable(node_id, status_text) for node status updates
        reset_buffer: if True, clears buffer_store (Full Run mode)
        """
        self.engine = engine
        self.log = log_fn or (lambda t, c="#89B4FA": None)
        self.status = status_fn or (lambda nid, s: None)

        if reset_buffer:
            GraphExecutor.buffer_store = []

        # Parse graph
        self.nodes = {n["id"]: n for n in graph_data["nodes"]}
        self.adj = defaultdict(list)       # from_id -> [(to_id, out_port, in_port)]
        self.in_edges = defaultdict(list)   # to_id -> [(from_id, out_port, in_port)]

        for e in graph_data["edges"]:
            self.adj[e["from"]].append((e["to"], e["fp"], e["tp"]))
            self.in_edges[e["to"]].append((e["from"], e["fp"], e["tp"]))

        # Node configs extracted from visual graph
        self.node_configs = {}  # node_id -> config dict from GUI widgets

        # Runtime data bus: node_id -> {port_index: data}
        self.output_data = {}
        self.skipped = set()  # node IDs to skip (e.g. cache hit short-circuit)

    def set_node_config(self, node_id, config):
        """Set runtime config for a node (k value, routing type, etc.)."""
        self.node_configs[node_id] = config

    def execute(self, query_text):
        """Run the full graph. Returns the final output text."""
        start = time.time()
        self.log(f"Graph Executor: starting with query '{query_text}'", "#A6E3A1")

        # Topological sort using Kahn's algorithm
        from node_editor import NODE_SPEC
        in_degree = {nid: 0 for nid in self.nodes}
        for nid in self.nodes:
            for (src, op, ip) in self.in_edges[nid]:
                if src in self.nodes:
                    in_degree[nid] += 1

        queue = sorted(
            [nid for nid, d in in_degree.items() if d == 0],
            key=lambda nid: NODE_SPEC.get(self.nodes[nid]["type"], {}).get("priority", 50)
        )

        result_text = ""
        levels_executed = 0

        while queue:
            levels_executed += 1
            # Execute all nodes at this level (potentially in parallel)
            next_queue = []
            level_nodes = [nid for nid in queue if nid not in self.skipped]

            if len(level_nodes) > 1:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {pool.submit(self._exec_node, nid, query_text): nid for nid in level_nodes}
                    for f in futures:
                        f.result()  # wait for all
            else:
                for nid in level_nodes:
                    self._exec_node(nid, query_text)

            # Advance
            for nid in queue:
                for (child_id, out_port, in_port) in self.adj.get(nid, []):
                    if child_id in in_degree:
                        in_degree[child_id] -= 1
                        if in_degree[child_id] == 0:
                            next_queue.append(child_id)

            queue = sorted(
                next_queue,
                key=lambda nid: NODE_SPEC.get(self.nodes[nid]["type"], {}).get("priority", 50)
            )

        # Collect final output from Response Output nodes
        for nid, n in self.nodes.items():
            if n["type"] == "Response Output" and nid in self.output_data:
                for port, data in self.output_data[nid].items():
                    if data:
                        result_text = data

        elapsed = round(time.time() - start, 2)
        self.log(f"Graph execution complete in {elapsed}s ({levels_executed} levels)", "#A6E3A1")
        return result_text

    def _get_input_data(self, node_id):
        """Gather all input data arriving at this node from upstream wires."""
        inputs = {}
        for (src_id, out_port, in_port) in self.in_edges.get(node_id, []):
            if src_id in self.output_data and out_port in self.output_data[src_id]:
                data = self.output_data[src_id][out_port]
                if data is not None:
                    inputs[in_port] = inputs.get(in_port, "") + str(data)
        return inputs

    def _set_output(self, node_id, port, data):
        if node_id not in self.output_data:
            self.output_data[node_id] = {}
        self.output_data[node_id][port] = data

    def _exec_node(self, nid, query_text):
        """Execute a single node based on its type."""
        n = self.nodes[nid]
        ntype = n["type"]
        cfg = self.node_configs.get(nid, {})
        inputs = self._get_input_data(nid)

        self.status(nid, "\U0001f7e2 Running")
        self.log(f"  [{ntype}] executing...", "#CBA6F7")

        try:
            if ntype == "Query Input":
                self._set_output(nid, 0, query_text)
                self.log(f"  [{ntype}] emitting query: '{query_text[:50]}...'", "#FAB387")

            elif ntype == "PDF Loader":
                # PDF loading is done beforehand via the UI button.
                # This node just passes through a signal that docs are loaded.
                self._set_output(nid, 0, "[docs_loaded]")

            elif ntype == "Cache":
                q = inputs.get(0, query_text)
                if q in GraphExecutor.cache_store:
                    # HIT → output port 0 (hit), skip miss branch
                    cached = GraphExecutor.cache_store[q]
                    self._set_output(nid, 0, cached)
                    self.log(f"  [Cache] HIT for '{q[:40]}...'", "#A6E3A1")
                    # Short-circuit: skip all nodes reachable ONLY through miss port (port 1)
                    self._short_circuit_port(nid, 1)
                else:
                    # MISS → output port 1 (miss), skip hit branch
                    self._set_output(nid, 1, q)
                    self.log(f"  [Cache] MISS for '{q[:40]}...'", "#F9E2AF")
                    self._short_circuit_port(nid, 0)

            elif ntype == "Router":
                data = inputs.get(0, "")
                rt = cfg.get("routing_type", "keyword")
                keyword = cfg.get("keyword", "")
                if rt == "keyword" and keyword and keyword.lower() in str(data).lower():
                    self._set_output(nid, 0, data)  # route_a
                    self._short_circuit_port(nid, 1)
                    self.log(f"  [Router] keyword '{keyword}' matched → route_a", "#F5C2E7")
                else:
                    self._set_output(nid, 1, data)  # route_b
                    self._short_circuit_port(nid, 0)
                    self.log(f"  [Router] → route_b (default)", "#F5C2E7")

            elif ntype == "FAISS DB":
                q = inputs.get(0, query_text)
                k = cfg.get("k", 4)
                if not self.engine.vector_db:
                    self.log(f"  [FAISS DB] WARNING: No index loaded! Index your PDFs first.", "#F38BA8")
                    self._set_output(nid, 0, "[NO_DB]")
                else:
                    docs = self.engine.retrieve(str(q), k=k)
                    context = "\n\n".join(doc.page_content for doc in docs)
                    self._set_output(nid, 0, context)
                    self.log(f"  [FAISS DB] retrieved {len(docs)} chunks (k={k})", "#89B4FA")

            elif ntype == "Buffer":
                data = inputs.get(0, "")
                window = cfg.get("window_size", 5)
                n_entries_before = len(GraphExecutor.buffer_store)
                if data and data not in ("", "[docs_loaded]"):
                    GraphExecutor.buffer_store.append(str(data))
                memory = "\n".join(GraphExecutor.buffer_store[-window:])
                self._set_output(nid, 0, memory if memory else "[no_history]")
                n_entries = len(GraphExecutor.buffer_store)
                if n_entries_before == 0:
                    self.log(f"  [Buffer] empty (first run — history builds after LLM responds)", "#F9E2AF")
                else:
                    self.log(f"  [Buffer] {n_entries} entries in memory (window={window})", "#F9E2AF")

            elif ntype == "Merge":
                # Combine all inputs
                parts = [str(inputs.get(p, "")) for p in sorted(inputs.keys())]
                merged = "\n\n---\n\n".join(p for p in parts if p)
                self._set_output(nid, 0, merged)
                self.log(f"  [Merge] combined {len(parts)} inputs", "#94E2D5")

            elif ntype == "Ollama LLM":
                prompt_text = inputs.get(0, "")
                sys_prompt = self.engine.system_prompt.replace("{context}", str(prompt_text))
                answer = self.engine.generate(f"{sys_prompt}\n\nQuestion: {query_text}")
                self._set_output(nid, 0, answer)
                # Store in cache for future hits
                GraphExecutor.cache_store[query_text] = answer
                self.log(f"  [Ollama LLM] generated {len(answer)} chars", "#CBA6F7")

            elif ntype == "Copy":
                data = inputs.get(0, "")
                self._set_output(nid, 0, data)
                self._set_output(nid, 1, data)
                self.log(f"  [Copy] duplicated to 2 outputs", "#CDD6F4")

            elif ntype == "Response Output":
                data = inputs.get(0, "")
                self._set_output(nid, 0, data)
                self.log(f"  [Response Output] received {len(str(data))} chars", "#F38BA8")

            # ── NEW NODES ──────────────────────────────────────────────────
            elif ntype == "Prompt Template":
                q       = str(inputs.get(0, query_text))
                context = str(inputs.get(1, ""))
                memory  = str(inputs.get(2, ""))
                template = cfg.get("template", "")
                if not template.strip():
                    template = (
                        "You are a helpful assistant.\n"
                        "Context:\n{context}\n\n"
                        "Conversation:\n{memory}\n\n"
                        "Question:\n{query}"
                    )
                prompt = template.replace("{query}", q)\
                                 .replace("{context}", context)\
                                 .replace("{memory}", memory)
                self._set_output(nid, 0, prompt)
                self.log(f"  [Prompt Template] built prompt ({len(prompt)} chars)", "#B4BEFE")

            elif ntype == "System Message":
                message = cfg.get("message", "You are a knowledgeable, concise, and helpful AI assistant.")
                self._set_output(nid, 0, message)
                self.log(f"  [System Message] emitting ({len(message)} chars)", "#B4BEFE")

            elif ntype == "Reranker":
                raw = inputs.get(0, "")
                top_k = cfg.get("top_k", 3)
                # Chunks are strings; score by length as heuristic if no score annotation.
                # If a chunk has [score=X] prefix injected by Top-K Retriever, parse it.
                def _score(chunk_str):
                    s = str(chunk_str)
                    if s.startswith("[score="):
                        try: return float(s.split("]")[0].replace("[score=", ""))
                        except: pass
                    return len(s)  # heuristic: longer = richer
                chunks = [c.strip() for c in str(raw).split("\n\n") if c.strip()]
                chunks.sort(key=_score, reverse=True)
                reranked = "\n\n".join(chunks[:top_k])
                self._set_output(nid, 0, reranked)
                self.log(f"  [Reranker] kept top {top_k} of {len(chunks)} chunks", "#74C7EC")

            elif ntype == "Memory Formatter":
                memory  = str(inputs.get(0, ""))
                max_tok = cfg.get("max_tokens", 500)
                # Estimate tokens as words/0.75
                max_words = int(max_tok * 0.75)
                words = memory.split()
                if len(words) > max_words:
                    memory = " ".join(words[-max_words:])
                    self.log(f"  [Memory Formatter] truncated to ~{max_tok} tokens", "#F9E2AF")
                else:
                    self.log(f"  [Memory Formatter] memory within limit ({len(words)} words)", "#F9E2AF")
                self._set_output(nid, 0, memory)

            elif ntype == "Conversation Starter":
                text = cfg.get("text", "")
                self._set_output(nid, 0, text)
                self.log(f"  [Conversation Starter] emitting starter ({len(text)} chars)", "#F9E2AF")

            elif ntype == "Score Filter":
                raw       = inputs.get(0, "")
                threshold = float(cfg.get("threshold", 0.5))
                chunks    = [c.strip() for c in str(raw).split("\n\n") if c.strip()]
                kept = []
                for chunk in chunks:
                    score = 1.0  # default if no annotation
                    if chunk.startswith("[score="):
                        try:
                            score = float(chunk.split("]")[0].replace("[score=", ""))
                        except: pass
                    if score >= threshold:
                        kept.append(chunk)
                filtered = "\n\n".join(kept)
                self._set_output(nid, 0, filtered)
                self.log(f"  [Score Filter] {len(kept)}/{len(chunks)} chunks passed (threshold={threshold})", "#74C7EC")

            elif ntype == "Debug Inspector":
                data = inputs.get(0, "")
                self._set_output(nid, 0, data)   # pure pass-through
                self.status(nid, "\U0001f7e4 Inspect")
                self.log(f"  [Debug Inspector] data: {str(data)[:120]}...", "#A6ADC8")
                # Feed visual widget from GUI thread via status signal (reuses nid)
                try:
                    # Emit a secondary status with the data payload for the UI
                    self.status(nid, f"\U0001f50e {str(data)[:80]}")
                except: pass

            elif ntype == "Multi Merge":
                # Collect all connected input ports in sorted order
                parts = []
                for port in sorted(inputs.keys()):
                    val = str(inputs.get(port, ""))
                    if val:
                        label = n.get("in_labels", [f"in_{port}"])
                        lbl   = label[port] if port < len(label) else f"port_{port}"
                        parts.append(f"[{lbl}]\n{val}")
                merged = "\n\n---\n\n".join(parts)
                self._set_output(nid, 0, merged)
                self.log(f"  [Multi Merge] merged {len(parts)} inputs", "#94E2D5")

            elif ntype == "Top-K Retriever":
                q = str(inputs.get(0, query_text))
                k = cfg.get("k", 5)
                if not self.engine.vector_db:
                    self.log(f"  [Top-K Retriever] WARNING: No FAISS index loaded!", "#F38BA8")
                    self._set_output(nid, 0, "[NO_DB]")
                else:
                    docs  = self.engine.retrieve(q, k=k)
                    # Annotate each chunk with a mock score for Reranker compatibility
                    parts = []
                    for i, doc in enumerate(docs):
                        score = round(1.0 - i * 0.05, 3)   # simulated descending score
                        parts.append(f"[score={score}]\n{doc.page_content}")
                    chunks = "\n\n".join(parts)
                    self._set_output(nid, 0, chunks)
                    self.log(f"  [Top-K Retriever] retrieved {len(docs)} docs (k={k})", "#74C7EC")

            elif ntype == "Seed Buffer Loader":
                seed = cfg.get("seed", "").strip()
                if seed and seed not in GraphExecutor.buffer_store:
                    for line in seed.split("\n"):
                        line = line.strip()
                        if line:
                            GraphExecutor.buffer_store.insert(0, line)
                    self.log(f"  [Seed Buffer Loader] injected {len(seed.splitlines())} seed turns", "#F9E2AF")
                else:
                    self.log(f"  [Seed Buffer Loader] seed already in buffer (skipped)", "#F9E2AF")
                # Output the seeds as a memory string
                self._set_output(nid, 0, seed)

            self.status(nid, "\U0001f7e2 Done")

        except Exception as e:
            self.status(nid, f"\U0001f534 Error: {e}")
            self.log(f"  [{ntype}] ERROR: {e}", "#F38BA8")


    def _short_circuit_port(self, node_id, port_to_skip):
        """Skip nodes reachable ONLY through the dead port. Nodes with other live inputs survive."""
        # Mark the specific port as dead (no data will flow from it)
        # We only need to mark direct targets for potential skipping
        dead_targets = set()
        for (child_id, out_port, in_port) in self.adj.get(node_id, []):
            if out_port == port_to_skip:
                dead_targets.add(child_id)

        # For each candidate, check if it has ANY other live incoming edge
        # (from a node that is NOT skipped and NOT from the dead port)
        actually_skipped = set()
        frontier = list(dead_targets)
        visited = set()

        while frontier:
            nid = frontier.pop(0)
            if nid in visited:
                continue
            visited.add(nid)

            # Check: does this node have ANY input from a live (non-skipped) source?
            has_live_input = False
            for (src_id, src_out_port, _) in self.in_edges.get(nid, []):
                if src_id in self.skipped or src_id in actually_skipped:
                    continue  # source is dead
                if src_id == node_id and src_out_port == port_to_skip:
                    continue  # this IS the dead port
                # This source is alive and feeding this node
                has_live_input = True
                break

            if has_live_input:
                continue  # Don't skip — it has another live input path

            actually_skipped.add(nid)
            self.skipped.add(nid)
            # Propagate: check children of this skipped node too
            for (child_id, op, ip) in self.adj.get(nid, []):
                frontier.append(child_id)

        if actually_skipped:
            names = [self.nodes[nid]["type"] for nid in actually_skipped if nid in self.nodes]
            self.log(f"  [Short-circuit] skipping: {', '.join(names)}", "#F9E2AF")
