"""
examples_tab.py  —  RAG Node Examples Gallery

A self-contained QWidget tab that:
  - Shows 10 worked examples (one per major architecture / node feature)
  - Lets the user read the description, inspect the node diagram
  - Provides a "Load into Canvas" button that serialises a ready-made graph
    and calls back into the main window to load it
"""
from __future__ import annotations
from typing import Callable, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QTextEdit, QSplitter, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QFont

from node_editor import Theme


# ─────────────────────────────────────────────────────────────────────────────
# Example definitions
# Each dict contains:
#   title       – Short display title
#   emoji       – Icon badge
#   category    – Colour-coded tag
#   tagline     – One-line subtitle
#   nodes_used  – List of node names shown in the "Nodes" chip row
#   description – Long multiline markdown-style explanation
#   diagram     – ASCII call-graph shown in the diagram pane
#   graph       – Serialised {nodes, edges} ready for NodeScene.deserialize()
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES: List[dict] = [

    # ── 1 ──────────────────────────────────────────────────────────────────
    {
        "title": "1 · Simple RAG",
        "emoji": "🟢",
        "category": "Retrieval",
        "tagline": "Classic Retrieve-then-Generate pipeline",
        "nodes_used": ["PDF Loader", "FAISS DB", "Query Input", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
The simplest possible RAG architecture. A PDF is indexed into a FAISS
vector database; the user's query retrieves relevant chunks; the LLM
composes an answer grounded in those chunks.

HOW IT WORKS
────────────
1. PDF Loader   → loads documents, signals FAISS DB to be ready.
2. Query Input  → user types a question.
3. FAISS DB     → similarity-searches the index (default k=4 chunks).
4. Ollama LLM   → receives the chunks as context and generates an answer.
5. Response Output → displays the final answer.

KEY VARIABLES
─────────────
• FAISS k      — controls how many chunks are retrieved (more = richer
                 context but slower generation).

WHEN TO USE
───────────
Starting point for any document QA task. Fast to set up. No memory,
no caching — every query goes to the LLM fresh.

TIP: Change k on the FAISS DB node from 4 → 8 to see how richer
context changes the LLM answer quality.
""",
        "diagram": """\
[PDF Loader] ──────────────────────► [FAISS DB]
                                         │ (k chunks)
[Query Input] ──────────────────────►   ▼
                                     [Ollama LLM]
                                         │
                                         ▼
                                  [Response Output]
""",
        "graph": {
            "nodes": [
                {"id": "ex1_pdf",  "type": "PDF Loader",      "x": -550, "y": 0,    "collapsed": False},
                {"id": "ex1_db",   "type": "FAISS DB",         "x": -200, "y": 0,    "collapsed": False},
                {"id": "ex1_q",    "type": "Query Input",      "x":  150, "y": -120, "collapsed": False},
                {"id": "ex1_llm",  "type": "Ollama LLM",       "x":  500, "y": -120, "collapsed": False},
                {"id": "ex1_out",  "type": "Response Output",  "x":  500, "y":  150, "collapsed": False},
            ],
            "edges": [
                {"from": "ex1_pdf", "fp": 0, "to": "ex1_db",  "tp": 0},
                {"from": "ex1_db",  "fp": 0, "to": "ex1_llm", "tp": 0},
                {"from": "ex1_q",   "fp": 0, "to": "ex1_llm", "tp": 0},
                {"from": "ex1_llm", "fp": 0, "to": "ex1_out", "tp": 0},
            ]
        }
    },

    # ── 2 ──────────────────────────────────────────────────────────────────
    {
        "title": "2 · Cache-Accelerated RAG",
        "emoji": "⚡",
        "category": "Caching",
        "tagline": "Skip the LLM entirely for repeated queries",
        "nodes_used": ["Query Input", "Cache", "FAISS DB", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
Adds a semantic cache layer before FAISS. Identical queries skip
FAISS + LLM entirely and return instantly from memory.

HOW IT WORKS
────────────
1. Query Input  → emits the query string.
2. Cache        → checks its in-memory dict.
     • HIT  (port 0) → answer goes directly to Response Output.
     • MISS (port 1) → query continues to FAISS → LLM.
3. FAISS DB     → retrieves relevant chunks (miss path only).
4. Ollama LLM   → generates answer, stores it in cache automatically.
5. Response Output → receives answer from whichever path won.

KEY VARIABLES
─────────────
• Cache TTL      — how long cache entries live (seconds).
• Cache Max      — maximum number of stored queries.

WHEN TO USE
───────────
Production systems with repeated queries (FAQ bots, help desks).
Cache hit rate improves dramatically after warm-up.

TIP: Ask the same question twice. Watch the terminal — on the second
run you'll see "[Cache] HIT" and neither FAISS nor LLM get called.
""",
        "diagram": """\
[Query Input]
     │
     ▼
  [Cache]
  /     \\
HIT     MISS
 │         │
 │     [FAISS DB]
 │         │
 │    [Ollama LLM]
  \\       /
   ▼     ▼
[Response Output]
""",
        "graph": {
            "nodes": [
                {"id": "ex2_q",    "type": "Query Input",      "x": 150,  "y": -200, "collapsed": False},
                {"id": "ex2_c",    "type": "Cache",             "x": 150,  "y": -20,  "collapsed": False},
                {"id": "ex2_db",   "type": "FAISS DB",          "x": -100, "y":  180, "collapsed": False},
                {"id": "ex2_llm",  "type": "Ollama LLM",        "x":  300, "y":  180, "collapsed": False},
                {"id": "ex2_out",  "type": "Response Output",   "x":  300, "y":  400, "collapsed": False},
            ],
            "edges": [
                {"from": "ex2_q",   "fp": 0, "to": "ex2_c",   "tp": 0},
                {"from": "ex2_c",   "fp": 0, "to": "ex2_out",  "tp": 0},  # cache hit
                {"from": "ex2_c",   "fp": 1, "to": "ex2_db",   "tp": 0},  # cache miss
                {"from": "ex2_db",  "fp": 0, "to": "ex2_llm",  "tp": 0},
                {"from": "ex2_llm", "fp": 0, "to": "ex2_out",  "tp": 0},
            ]
        }
    },

    # ── 3 ──────────────────────────────────────────────────────────────────
    {
        "title": "3 · Conversational Memory RAG",
        "emoji": "🧠",
        "category": "Memory",
        "tagline": "Multi-turn chat with sliding-window history",
        "nodes_used": ["Query Input", "Buffer", "FAISS DB", "Merge", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
Extends Simple RAG with a sliding-window conversation buffer so the
LLM can answer follow-up questions coherently ("tell me more",
"who is she?").

HOW IT WORKS
────────────
1. Query Input  → the new user query.
2. Buffer       → stores the last N query-answer pairs (window_size).
3. FAISS DB     → retrieves document chunks for the current query.
4. Merge        → combines [FAISS chunks] + [conversation history].
5. Ollama LLM   → receives the merged context and answers.
6. Response Output → shows the answer.
   → The answer is also fed back into Buffer for the next turn.

KEY VARIABLES
─────────────
• Buffer window_size — how many past turns to keep (default 5).
  Set to 1 for "last Q only", 10 for long conversations.

WHEN TO USE
───────────
Chatbots, interview assistants, document exploration over several
turns. The bigger the window, the more context the LLM has.

TIP: Run three questions in sequence. On the third, refer back to
the first answer by pronoun and watch the LLM resolve it correctly
because the buffer contains all prior turns.
""",
        "diagram": """\
[Query Input] ─────────────────────────────► [Buffer]
     │                                           │ (last N turns)
     ▼                                           ▼
[FAISS DB] ──────────────────────────────► [Merge]
               (k chunks)                      │
                                               ▼
                                         [Ollama LLM]
                                               │
                                               ▼
                                       [Response Output]
""",
        "graph": {
            "nodes": [
                {"id": "ex3_q",    "type": "Query Input",      "x":  -200, "y": -150, "collapsed": False},
                {"id": "ex3_buf",  "type": "Buffer",            "x":   200, "y": -150, "collapsed": False},
                {"id": "ex3_db",   "type": "FAISS DB",          "x":  -200, "y":   50, "collapsed": False},
                {"id": "ex3_mrg",  "type": "Merge",             "x":   200, "y":   50, "collapsed": False},
                {"id": "ex3_llm",  "type": "Ollama LLM",        "x":   200, "y":  250, "collapsed": False},
                {"id": "ex3_out",  "type": "Response Output",   "x":   200, "y":  430, "collapsed": False},
            ],
            "edges": [
                {"from": "ex3_q",   "fp": 0, "to": "ex3_buf", "tp": 0},
                {"from": "ex3_q",   "fp": 0, "to": "ex3_db",  "tp": 0},
                {"from": "ex3_buf", "fp": 0, "to": "ex3_mrg", "tp": 1},
                {"from": "ex3_db",  "fp": 0, "to": "ex3_mrg", "tp": 0},
                {"from": "ex3_mrg", "fp": 0, "to": "ex3_llm", "tp": 0},
                {"from": "ex3_llm", "fp": 0, "to": "ex3_out", "tp": 0},
            ]
        }
    },

    # ── 4 ──────────────────────────────────────────────────────────────────
    {
        "title": "4 · Prompt Engineering Pipeline",
        "emoji": "📝",
        "category": "Prompt",
        "tagline": "Fine-tune exact prompts with System Message + Prompt Template",
        "nodes_used": ["Query Input", "FAISS DB", "System Message", "Prompt Template", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
Gives you full control over what the LLM receives. Instead of the
built-in system prompt inside LLM node, you wire a System Message
and a custom Prompt Template to precisely format every request.

HOW IT WORKS
────────────
1. System Message    → a zero-input node emitting your system instruction
                       (e.g. "You are a terse legal analyst.").
2. Query Input       → the user question.
3. FAISS DB          → retrieved context chunks.
4. Prompt Template   → receives (query, context, memory), fills in
                       your custom template string, emits a clean prompt.
5. Ollama LLM        → receives the formatted prompt on port 0.
6. Response Output   → shows the result.

KEY VARIABLES
─────────────
• System Message text — changes LLM persona completely.
• Prompt Template     — edit the template; use {query}, {context},
                        {memory} as placeholders. Hit "Reset to Default"
                        to restore the built-in format.

WHEN TO USE
───────────
Expert-mode RAG. Use when default prompts give hallucinations or
wrong tone. Essential for domain-specific applications (legal, medical,
code review).

TIP: Try changing System Message to "Reply only in bullet points"
and run the same query. The entire response style changes.
""",
        "diagram": """\
[System Message]──────────────────────────────┐
                                               │
[Query Input] ──────────────────► [Prompt Template] ──► [Ollama LLM] ──► [Response Output]
                                         ▲
[FAISS DB] (k chunks) ───────────────────┘
""",
        "graph": {
            "nodes": [
                {"id": "ex4_sys",  "type": "System Message",    "x": -400, "y": -200, "collapsed": False},
                {"id": "ex4_q",    "type": "Query Input",        "x": -400, "y":    0, "collapsed": False},
                {"id": "ex4_db",   "type": "FAISS DB",           "x": -400, "y":  200, "collapsed": False},
                {"id": "ex4_pt",   "type": "Prompt Template",    "x":   50, "y":    0, "collapsed": False},
                {"id": "ex4_llm",  "type": "Ollama LLM",         "x":  450, "y":    0, "collapsed": False},
                {"id": "ex4_out",  "type": "Response Output",    "x":  450, "y":  200, "collapsed": False},
            ],
            "edges": [
                {"from": "ex4_q",   "fp": 0, "to": "ex4_db",  "tp": 0},
                {"from": "ex4_q",   "fp": 0, "to": "ex4_pt",  "tp": 0},  # query → template port 0
                {"from": "ex4_db",  "fp": 0, "to": "ex4_pt",  "tp": 1},  # chunks → template port 1 (context)
                {"from": "ex4_sys", "fp": 0, "to": "ex4_llm", "tp": 0},  # system → LLM port 0
                {"from": "ex4_pt",  "fp": 0, "to": "ex4_llm", "tp": 0},  # prompt → LLM (overrides sys)
                {"from": "ex4_llm", "fp": 0, "to": "ex4_out", "tp": 0},
            ]
        }
    },

    # ── 5 ──────────────────────────────────────────────────────────────────
    {
        "title": "5 · Reranking & Score Filtering",
        "emoji": "🎯",
        "category": "Retrieval",
        "tagline": "Top-K retrieval → score filter → reranker → LLM",
        "nodes_used": ["Query Input", "Top-K Retriever", "Score Filter", "Reranker", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
High-quality retrieval pipeline. Retrieves more chunks than needed,
filters by relevance score, then reranks to keep only the very best
before sending to the LLM. Reduces hallucination and noise.

HOW IT WORKS
────────────
1. Top-K Retriever   → calls FAISS with a large k (e.g. 10).
                       Annotates each chunk with [score=X] for downstream
                       nodes to read.
2. Score Filter      → drops chunks scoring below threshold (e.g. 0.5).
3. Reranker          → sorts survivors by score, keeps top_k (e.g. 3).
4. Ollama LLM        → only sees the top-3 highest-confidence chunks.
5. Response Output   → final answer.

KEY VARIABLES
─────────────
• Top-K Retriever k      — how many candidates to fetch (start wide).
• Score Filter threshold — minimum relevance (0 = keep all, 1 = perfect).
• Reranker top_k         — final number of chunks to pass to LLM.

WHEN TO USE
───────────
When simple k=4 retrieval returns irrelevant chunks. Medical or legal
RAG where wrong context is dangerous. Long documents with many topics.

TIP: Set k=10, threshold=0.6, top_k=3 and compare LLM answers against
the Simple RAG example — you'll get more focused, accurate responses.
""",
        "diagram": """\
[Query Input]
     │
     ▼
[Top-K Retriever] (k=10, annotates scores)
     │
     ▼
[Score Filter] (keep score ≥ 0.5)
     │
     ▼
[Reranker]  (keep top 3)
     │
     ▼
[Ollama LLM]
     │
     ▼
[Response Output]
""",
        "graph": {
            "nodes": [
                {"id": "ex5_q",    "type": "Query Input",       "x":  0,   "y": -200, "collapsed": False},
                {"id": "ex5_tk",   "type": "Top-K Retriever",   "x":  0,   "y":  -30, "collapsed": False},
                {"id": "ex5_sf",   "type": "Score Filter",      "x":  0,   "y":  150, "collapsed": False},
                {"id": "ex5_rr",   "type": "Reranker",          "x":  0,   "y":  320, "collapsed": False},
                {"id": "ex5_llm",  "type": "Ollama LLM",        "x":  0,   "y":  490, "collapsed": False},
                {"id": "ex5_out",  "type": "Response Output",   "x":  0,   "y":  660, "collapsed": False},
            ],
            "edges": [
                {"from": "ex5_q",   "fp": 0, "to": "ex5_tk",  "tp": 0},
                {"from": "ex5_tk",  "fp": 0, "to": "ex5_sf",  "tp": 0},
                {"from": "ex5_sf",  "fp": 0, "to": "ex5_rr",  "tp": 0},
                {"from": "ex5_rr",  "fp": 0, "to": "ex5_llm", "tp": 0},
                {"from": "ex5_llm", "fp": 0, "to": "ex5_out", "tp": 0},
            ]
        }
    },

    # ── 6 ──────────────────────────────────────────────────────────────────
    {
        "title": "6 · Conditional Router RAG",
        "emoji": "🔀",
        "category": "Logic",
        "tagline": "Keyword-based routing to specialized LLM paths",
        "nodes_used": ["Query Input", "Router", "FAISS DB", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
Route queries to different specialist pipelines based on content.
"Tell me about Python" → technical LLM path.
"What is the refund policy?" → customer-service LLM path.

HOW IT WORKS
────────────
1. Query Input  → the user question.
2. Router       → inspects the query for a keyword.
     • Port 0 (route_a) — keyword matched → specialist path.
     • Port 1 (route_b) — no match → general fallback path.
3. Each path has its own Ollama LLM + Response Output.
4. The inactive path is short-circuited (never runs).

KEY VARIABLES
─────────────
• Router → Type:    "keyword" (string match) or "confidence".
• Router → Keyword: the trigger word (case-insensitive).

WHEN TO USE
───────────
Multi-domain chatbots. Different LLM system prompts per domain.
A/B testing two different prompting strategies.

TIP: Set keyword="technical" and ask "explain this technical principle".
Then change keyword to something else and ask the same — watch which
LLM runs via the terminal.
""",
        "diagram": """\
[Query Input]
     │
  [Router] (keyword="technical")
   /     \\
route_a  route_b
  │         │
[LLM A]  [LLM B]
  │         │
[Out A]  [Out B]
""",
        "graph": {
            "nodes": [
                {"id": "ex6_q",    "type": "Query Input",      "x":  0,   "y": -200, "collapsed": False},
                {"id": "ex6_rt",   "type": "Router",            "x":  0,   "y":  -30, "collapsed": False},
                {"id": "ex6_la",   "type": "Ollama LLM",        "x": -300, "y":  180, "collapsed": False},
                {"id": "ex6_lb",   "type": "Ollama LLM",        "x":  300, "y":  180, "collapsed": False},
                {"id": "ex6_oa",   "type": "Response Output",   "x": -300, "y":  380, "collapsed": False},
                {"id": "ex6_ob",   "type": "Response Output",   "x":  300, "y":  380, "collapsed": False},
            ],
            "edges": [
                {"from": "ex6_q",  "fp": 0, "to": "ex6_rt", "tp": 0},
                {"from": "ex6_rt", "fp": 0, "to": "ex6_la", "tp": 0},
                {"from": "ex6_rt", "fp": 1, "to": "ex6_lb", "tp": 0},
                {"from": "ex6_la", "fp": 0, "to": "ex6_oa", "tp": 0},
                {"from": "ex6_lb", "fp": 0, "to": "ex6_ob", "tp": 0},
            ]
        }
    },

    # ── 7 ──────────────────────────────────────────────────────────────────
    {
        "title": "7 · Seeded Conversation RAG",
        "emoji": "🌱",
        "category": "Memory",
        "tagline": "Pre-load context so LLM knows the domain before Q1",
        "nodes_used": ["Seed Buffer Loader", "Conversation Starter", "Query Input", "FAISS DB", "Memory Formatter", "Merge", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
Primes the conversation with known background BEFORE the first user
query. Perfect for onboarding scenarios where you want the LLM to
already "know" the customer's name, account tier, or session context.

HOW IT WORKS
────────────
1. Seed Buffer Loader  → injects multi-line seed text into the shared
                         conversation buffer at graph start (priority 0).
2. Conversation Starter → emits a one-off greeting/persona line as the
                          first memory entry.
3. Query Input          → the real user question.
4. FAISS DB             → document retrieval.
5. Memory Formatter     → truncates buffer to max_tokens before LLM.
6. Merge                → combines [chunks] + [formatted memory].
7. Ollama LLM           → answers with full context.
8. Response Output      → displays answer.

KEY VARIABLES
─────────────
• Seed Buffer Loader seed  — paste background facts (one per line).
• Conversation Starter text — the opening persona/greeting line.
• Memory Formatter max_tokens — control context window budget.

TIP: In the Seed Buffer Loader, write:
  "Customer name: Alice"
  "Account tier: Premium"
  "Previous issue: billing error in March"
Then ask "what do you know about me?" and see the LLM answer with
the seeded facts even though you never typed them in the query.
""",
        "diagram": """\
[Seed Buffer Loader] ─────► (buffer_store)
[Conversation Starter] ────► (memory)
                                   │
[Query Input] ─────────────► [FAISS DB]
     │                           │
     │                    [Memory Formatter]
     │                           │
     └───────────────────► [Merge]
                               │
                          [Ollama LLM]
                               │
                        [Response Output]
""",
        "graph": {
            "nodes": [
                {"id": "ex7_sb",  "type": "Seed Buffer Loader",  "x": -550, "y": -200, "collapsed": False},
                {"id": "ex7_cs",  "type": "Conversation Starter","x": -550, "y":    0, "collapsed": False},
                {"id": "ex7_q",   "type": "Query Input",          "x":  -50, "y": -200, "collapsed": False},
                {"id": "ex7_db",  "type": "FAISS DB",             "x":  -50, "y":   50, "collapsed": False},
                {"id": "ex7_mf",  "type": "Memory Formatter",     "x":  300, "y": -100, "collapsed": False},
                {"id": "ex7_mrg", "type": "Merge",                "x":  300, "y":  150, "collapsed": False},
                {"id": "ex7_llm", "type": "Ollama LLM",           "x":  300, "y":  350, "collapsed": False},
                {"id": "ex7_out", "type": "Response Output",      "x":  300, "y":  540, "collapsed": False},
            ],
            "edges": [
                {"from": "ex7_cs",  "fp": 0, "to": "ex7_mf",  "tp": 0},
                {"from": "ex7_q",   "fp": 0, "to": "ex7_db",  "tp": 0},
                {"from": "ex7_db",  "fp": 0, "to": "ex7_mrg", "tp": 0},
                {"from": "ex7_mf",  "fp": 0, "to": "ex7_mrg", "tp": 1},
                {"from": "ex7_mrg", "fp": 0, "to": "ex7_llm", "tp": 0},
                {"from": "ex7_llm", "fp": 0, "to": "ex7_out", "tp": 0},
            ]
        }
    },

    # ── 8 ──────────────────────────────────────────────────────────────────
    {
        "title": "8 · Multi-Source Fusion",
        "emoji": "🔗",
        "category": "Processor",
        "tagline": "Combine system message + memory + chunks via Multi Merge",
        "nodes_used": ["System Message", "Query Input", "Buffer", "Top-K Retriever", "Multi Merge", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
Demonstrates the Multi Merge node which can combine up to 4 named
inputs. Here: [system instruction] + [conversation history] +
[retrieved document chunks] all flow into one merged prompt that
the LLM receives as a single structured input.

HOW IT WORKS
────────────
1. System Message   → persona/instructions (zero-input source node).
2. Buffer           → rolling conversation history.
3. Top-K Retriever  → similarity search on the user query.
4. Multi Merge      → combines all three inputs with section labels:
                      [in_0] system / [in_1] memory / [in_2] context.
5. Ollama LLM       → receives the labelled, structured prompt.
6. Response Output  → final answer.

KEY VARIABLES
─────────────
• System Message content — controls LLM role.
• Buffer window_size      — length of memory included.
• Top-K Retriever k       — number of retrieved document chunks.

WHEN TO USE
───────────
When you need precise control over the layout of the full prompt
that the LLM sees. Multi Merge gives each section a named label
so you can debug exactly what the model received.
""",
        "diagram": """\
[System Message] ────────────────────────────┐
                                             │ port 0
[Buffer] (history) ──────────────────────── Multi Merge ─► [Ollama LLM] ─► [Response Output]
                                             │ port 1
[Top-K Retriever] (chunks) ─────────────────┘ port 2
     ▲
[Query Input]
""",
        "graph": {
            "nodes": [
                {"id": "ex8_sys",  "type": "System Message",    "x": -400, "y": -300, "collapsed": False},
                {"id": "ex8_buf",  "type": "Buffer",             "x": -400, "y": -100, "collapsed": False},
                {"id": "ex8_q",    "type": "Query Input",        "x": -400, "y":  100, "collapsed": False},
                {"id": "ex8_tk",   "type": "Top-K Retriever",    "x": -400, "y":  300, "collapsed": False},
                {"id": "ex8_mm",   "type": "Multi Merge",        "x":   50, "y":  -50, "collapsed": False},
                {"id": "ex8_llm",  "type": "Ollama LLM",         "x":  400, "y":  -50, "collapsed": False},
                {"id": "ex8_out",  "type": "Response Output",    "x":  400, "y":  150, "collapsed": False},
            ],
            "edges": [
                {"from": "ex8_q",   "fp": 0, "to": "ex8_buf", "tp": 0},
                {"from": "ex8_q",   "fp": 0, "to": "ex8_tk",  "tp": 0},
                {"from": "ex8_sys", "fp": 0, "to": "ex8_mm",  "tp": 0},
                {"from": "ex8_buf", "fp": 0, "to": "ex8_mm",  "tp": 1},
                {"from": "ex8_tk",  "fp": 0, "to": "ex8_mm",  "tp": 2},
                {"from": "ex8_mm",  "fp": 0, "to": "ex8_llm", "tp": 0},
                {"from": "ex8_llm", "fp": 0, "to": "ex8_out", "tp": 0},
            ]
        }
    },

    # ── 9 ──────────────────────────────────────────────────────────────────
    {
        "title": "9 · Debug-Instrumented Pipeline",
        "emoji": "🔍",
        "category": "Debug",
        "tagline": "Inspect data at every stage without blocking flow",
        "nodes_used": ["Query Input", "Top-K Retriever", "Debug Inspector", "Score Filter", "Reranker", "Prompt Template", "Debug Inspector", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
Shows how to use Debug Inspector as a non-blocking spy node.
Insert it anywhere in the graph to see exactly what data is flowing
through that wire — without changing the pipeline behaviour.

HOW IT WORKS
────────────
1. Query Input      → user question.
2. Top-K Retriever  → fetch k chunks (annotated with scores).
3. Debug Inspector  → FIRST spy: logs raw retrieved chunks.
4. Score Filter + Reranker → quality pipeline.
5. Debug Inspector  → SECOND spy: logs the post-filter chunks.
6. Prompt Template  → build structured prompt.
7. Ollama LLM       → generate answer.
8. Response Output  → display.

Debug Inspector simply logs its input, then passes it through
UNCHANGED. It is always priority 0, so it never blocks execution.

WHEN TO USE
───────────
Prototype debugging: "are my chunks relevant?", "is the filter too
aggressive?", "what exactly does the LLM see?". Remove inspectors
when done to slim the graph.

TIP: Open the Debug Inspector node widget during execution — the
text box shows the live data that flowed through that point.
""",
        "diagram": """\
[Query Input]
     │
[Top-K Retriever]
     │
[Debug Inspector #1]  ← "what did FAISS return?"
     │
[Score Filter]
     │
[Reranker]
     │
[Debug Inspector #2]  ← "what reaches the LLM?"
     │
[Prompt Template]
     │
[Ollama LLM]
     │
[Response Output]
""",
        "graph": {
            "nodes": [
                {"id": "ex9_q",    "type": "Query Input",       "x":  0, "y": -600, "collapsed": False},
                {"id": "ex9_tk",   "type": "Top-K Retriever",   "x":  0, "y": -440, "collapsed": False},
                {"id": "ex9_di1",  "type": "Debug Inspector",   "x":  0, "y": -280, "collapsed": False},
                {"id": "ex9_sf",   "type": "Score Filter",      "x":  0, "y": -120, "collapsed": False},
                {"id": "ex9_rr",   "type": "Reranker",          "x":  0, "y":   40, "collapsed": False},
                {"id": "ex9_di2",  "type": "Debug Inspector",   "x":  0, "y":  200, "collapsed": False},
                {"id": "ex9_pt",   "type": "Prompt Template",   "x":  0, "y":  360, "collapsed": False},
                {"id": "ex9_llm",  "type": "Ollama LLM",        "x":  0, "y":  540, "collapsed": False},
                {"id": "ex9_out",  "type": "Response Output",   "x":  0, "y":  710, "collapsed": False},
            ],
            "edges": [
                {"from": "ex9_q",   "fp": 0, "to": "ex9_tk",  "tp": 0},
                {"from": "ex9_tk",  "fp": 0, "to": "ex9_di1", "tp": 0},
                {"from": "ex9_di1", "fp": 0, "to": "ex9_sf",  "tp": 0},
                {"from": "ex9_sf",  "fp": 0, "to": "ex9_rr",  "tp": 0},
                {"from": "ex9_rr",  "fp": 0, "to": "ex9_di2", "tp": 0},
                {"from": "ex9_di2", "fp": 0, "to": "ex9_pt",  "tp": 0},
                {"from": "ex9_q",   "fp": 0, "to": "ex9_pt",  "tp": 0},  # query → template port 0
                {"from": "ex9_pt",  "fp": 0, "to": "ex9_llm", "tp": 0},
                {"from": "ex9_llm", "fp": 0, "to": "ex9_out", "tp": 0},
            ]
        }
    },

    # ── 10 ─────────────────────────────────────────────────────────────────
    {
        "title": "10 · Full Production RAG",
        "emoji": "🚀",
        "category": "Full Stack",
        "tagline": "Every node type working together in one graph",
        "nodes_used": ["Seed Buffer Loader", "System Message", "Query Input", "Cache", "Top-K Retriever",
                       "Score Filter", "Reranker", "Memory Formatter", "Multi Merge", "Ollama LLM", "Response Output"],
        "description": """\
PURPOSE
───────
The complete production-grade RAG system combining every feature:
caching, semantic retrieval with filtering, conversation memory,
seeded context, structured prompting, and graceful degradation.

HOW IT WORKS
────────────
 Start-of-graph (priority 0–1, zero-input sources):
   • Seed Buffer Loader  → injects domain background into buffer.
   • System Message      → emits the LLM persona.

 Query path:
   • Query Input → Cache (hit=instant answer / miss=full pipeline).

 Miss path (full computation):
   • Top-K Retriever → Score Filter → Reranker (quality chunks).
   • Buffer → Memory Formatter (trimmed history).
   • Multi Merge → combines [system] + [history] + [chunks].
   • Ollama LLM → generates answer.

 Output:
   • Response Output shows final answer.
   • Cache stores it for future hits.

WHAT TO OBSERVE
───────────────
1. First run → cache MISS, all nodes execute.
2. Same query again → cache HIT, Reranker, FAISS etc. all skipped.
3. Different query → cache MISS, full pipeline again.
4. Change Reranker top_k → fewer/better chunks reach LLM.
5. Edit Seed Buffer Loader → the LLM "remembers" new background.

This is the reference architecture for a production system.
""",
        "diagram": """\
[Seed Buffer Loader]──►(buffer)    [System Message]
                                         │ port 0
[Query Input]                            │
     │ port 0              ┌─────────────┤
  [Cache]                  │             │
   /   \\                   │         ▼
  HIT  MISS           [Memory     [Multi Merge] ◄─── [Reranker]
   │     │             Formatter]      │                  ▲
   │  [Top-K]             │            │            [Score Filter]
   │  [Score Filter]      │            │                  ▲
   │  [Reranker]          │            │            [Top-K Retriever]
   │     └────────────────┘            │
   │                                   ▼
   │                            [Ollama LLM]
   └──────────────────────────────────►▼
                               [Response Output]
""",
        "graph": {
            "nodes": [
                {"id": "ex10_sb",   "type": "Seed Buffer Loader",  "x": -700, "y": -300, "collapsed": False},
                {"id": "ex10_sys",  "type": "System Message",       "x": -700, "y": -100, "collapsed": False},
                {"id": "ex10_q",    "type": "Query Input",           "x": -200, "y": -300, "collapsed": False},
                {"id": "ex10_c",    "type": "Cache",                 "x": -200, "y": -100, "collapsed": False},
                {"id": "ex10_tk",   "type": "Top-K Retriever",       "x":   50, "y":  150, "collapsed": False},
                {"id": "ex10_sf",   "type": "Score Filter",          "x":   50, "y":  330, "collapsed": False},
                {"id": "ex10_rr",   "type": "Reranker",              "x":   50, "y":  510, "collapsed": False},
                {"id": "ex10_buf",  "type": "Buffer",                "x":  400, "y": -100, "collapsed": False},
                {"id": "ex10_mf",   "type": "Memory Formatter",      "x":  400, "y":  150, "collapsed": False},
                {"id": "ex10_mm",   "type": "Multi Merge",           "x":  400, "y":  380, "collapsed": False},
                {"id": "ex10_llm",  "type": "Ollama LLM",            "x":  400, "y":  580, "collapsed": False},
                {"id": "ex10_out",  "type": "Response Output",       "x":  400, "y":  760, "collapsed": False},
            ],
            "edges": [
                # Query → Cache
                {"from": "ex10_q",   "fp": 0, "to": "ex10_c",   "tp": 0},
                # Cache hit → output
                {"from": "ex10_c",   "fp": 0, "to": "ex10_out", "tp": 0},
                # Cache miss → retrieval chain
                {"from": "ex10_c",   "fp": 1, "to": "ex10_tk",  "tp": 0},
                {"from": "ex10_tk",  "fp": 0, "to": "ex10_sf",  "tp": 0},
                {"from": "ex10_sf",  "fp": 0, "to": "ex10_rr",  "tp": 0},
                # Query → buffer
                {"from": "ex10_q",   "fp": 0, "to": "ex10_buf", "tp": 0},
                {"from": "ex10_buf", "fp": 0, "to": "ex10_mf",  "tp": 0},
                # Multi Merge inputs
                {"from": "ex10_sys", "fp": 0, "to": "ex10_mm",  "tp": 0},
                {"from": "ex10_mf",  "fp": 0, "to": "ex10_mm",  "tp": 1},
                {"from": "ex10_rr",  "fp": 0, "to": "ex10_mm",  "tp": 2},
                # LLM + output
                {"from": "ex10_mm",  "fp": 0, "to": "ex10_llm", "tp": 0},
                {"from": "ex10_llm", "fp": 0, "to": "ex10_out", "tp": 0},
            ]
        }
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Category → colour map
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Retrieval":   ("#74C7EC", "#1E3A4A"),
    "Caching":     ("#89DCEB", "#1A3848"),
    "Memory":      ("#F9E2AF", "#4A3000"),
    "Prompt":      ("#B4BEFE", "#1A1050"),
    "Logic":       ("#F5C2E7", "#4A0030"),
    "Processor":   ("#94E2D5", "#003A30"),
    "Debug":       ("#A6ADC8", "#1A1E2A"),
    "Full Stack":  ("#A6E3A1", "#003010"),
}


# ─────────────────────────────────────────────────────────────────────────────
# ExamplesTab widget
# ─────────────────────────────────────────────────────────────────────────────
class ExamplesTab(QWidget):
    """
    Full gallery tab showing all RAG example architectures.

    load_callback(graph_dict) — called when the user clicks
    "Load into Canvas".  Should call window.load_from_dict(graph_dict).
    """

    def __init__(self, load_callback: Callable, back_callback: Callable, parent=None):
        super().__init__(parent)
        self._load_cb = load_callback
        self._back_cb = back_callback
        self._selected = 0
        self._build()

    # ── construction ─────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── LEFT: example list ────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(280)
        left.setObjectName("ex_sidebar")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 14, 12, 12)
        ll.setSpacing(6)

        hdr = QLabel("📚  RAG Examples")
        hdr.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        # Style hdr via setStyleSheet for better contrast
        hdr.setStyleSheet("color:#89b4fa; margin-bottom: 4px;")
        ll.addWidget(hdr)

        sub = QLabel("Click an example to explore.\nLoad it directly into the canvas.")
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#888;font-size:12px;margin-bottom:12px")
        ll.addWidget(sub)

        self._list = QListWidget()
        self._list.setSpacing(4)
        self._list.currentRowChanged.connect(self._on_select)
        ll.addWidget(self._list)

        for ex in EXAMPLES:
            cat = ex.get("category", "")
            fg, _ = CATEGORY_COLORS.get(cat, ("#CDD6F4", "#111"))
            item = QListWidgetItem(f" {ex['emoji']}  {ex['title']}")
            item.setForeground(QColor(fg))
            font = QFont("Segoe UI", 10)
            item.setFont(font)
            self._list.addItem(item)

        root.addWidget(left)

        # ── RIGHT: detail pane ────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # title bar
        self._title_bar = QWidget()
        self._title_bar.setFixedHeight(72)
        tbl = QHBoxLayout(self._title_bar)
        tbl.setContentsMargins(24, 0, 24, 0)

        self._lbl_title = QLabel()
        # Use a large font but ensure emoji scales well
        self._lbl_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        tbl.addWidget(self._lbl_title)

        tbl.addStretch()

        self._lbl_cat = QLabel()
        self._lbl_cat.setFixedHeight(28)
        self._lbl_cat.setMinimumWidth(100)
        self._lbl_cat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_cat.setStyleSheet(
            "border-radius:6px;padding:0 12px;font-weight:bold;font-size:12px")
        tbl.addWidget(self._lbl_cat)

        self._btn_back = QPushButton("\u21b0  Back to Builder")
        self._btn_back.setFixedHeight(36)
        self._btn_back.setStyleSheet(
            "background:#45475A;color:white;border:none;font-weight:bold;"
            "border-radius:8px;padding:0 20px;margin-left:16px;font-size:13px")
        self._btn_back.clicked.connect(self._back_cb)
        tbl.addWidget(self._btn_back)

        self._btn_load = QPushButton("⬆  Load into Canvas")
        self._btn_load.setFixedHeight(36)
        self._btn_load.setStyleSheet(
            "background:#10B981;color:white;border:none;font-weight:bold;"
            "border-radius:8px;padding:0 20px;margin-left:16px;font-size:13px")
        self._btn_load.clicked.connect(self._on_load)
        tbl.addWidget(self._btn_load)

        rl.addWidget(self._title_bar)

        # splitter: description | diagram
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(1) # Subtle splitter line

        # description (using QTextEdit as it naturally supports scrolling)
        self._desc = QTextEdit()
        self._desc.setReadOnly(True)
        self._desc.setFont(QFont("Cascadia Code", 10))
        self._desc.setStyleSheet("border:none; padding: 20px;")
        sp.addWidget(self._desc)

        # right sub-panel: tagline + "Nodes used" chips + diagram
        rpan = QWidget()
        rpan_l = QVBoxLayout(rpan)
        rpan_l.setContentsMargins(20, 20, 20, 20)
        rpan_l.setSpacing(14)

        self._lbl_tagline = QLabel()
        self._lbl_tagline.setWordWrap(True)
        self._lbl_tagline.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        self._lbl_tagline.setStyleSheet("color:#89b4fa; font-style: italic;")
        rpan_l.addWidget(self._lbl_tagline)

        chips_lbl = QLabel("Nodes used in this architecture:")
        chips_lbl.setStyleSheet("font-weight:bold;font-size:11px;color:#888; text-transform: uppercase;")
        rpan_l.addWidget(chips_lbl)

        # ── Scrollable Chips Row ──────────────────────────────────────
        self._chips_scroll = QScrollArea()
        self._chips_scroll.setWidgetResizable(True)
        self._chips_scroll.setFixedHeight(48)
        self._chips_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._chips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chips_scroll.setStyleSheet("background:transparent; border:none;")

        self._chips_row = QWidget()
        self._chips_row.setStyleSheet("background:transparent;")
        self._chips_layout = QHBoxLayout(self._chips_row)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(8)
        self._chips_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        self._chips_scroll.setWidget(self._chips_row)
        rpan_l.addWidget(self._chips_scroll)

        diag_lbl = QLabel("Topological Data Flow:")
        diag_lbl.setStyleSheet("font-weight:bold;font-size:11px;color:#888;margin-top:10px; text-transform: uppercase;")
        rpan_l.addWidget(diag_lbl)

        self._diagram = QTextEdit()
        self._diagram.setReadOnly(True)
        self._diagram.setFont(QFont("Cascadia Code", 10))
        self._diagram.setStyleSheet("border:1px solid #313244;border-radius:10px;padding:15px; background:#181825;")
        rpan_l.addWidget(self._diagram, 1)

        sp.addWidget(rpan)
        sp.setSizes([450, 450])
        rl.addWidget(sp, 1)

        root.addWidget(right, 1)

        # select first
        self._list.setCurrentRow(0)

    # ── selection ─────────────────────────────────────────────────────
    def _on_select(self, row: int):
        if row < 0 or row >= len(EXAMPLES):
            return
        self._selected = row
        ex = EXAMPLES[row]

        cat = ex.get("category", "")
        fg_col, bg_col = CATEGORY_COLORS.get(cat, ("#CDD6F4", "#1E2030"))

        # title bar - Clean & Transparent
        self._lbl_title.setText(f"{ex['emoji']}  {ex['title']}")
        self._title_bar.setStyleSheet(f"background: transparent; border-bottom: 2px solid {fg_col}33;")

        self._lbl_cat.setText(f"{cat.upper()}")
        self._lbl_cat.setStyleSheet(
            f"background:transparent; color:{fg_col}; border:1px solid {fg_col}; border-radius:6px;"
            f"padding:0 12px; font-weight:bold; font-size:10px; letter-spacing:1px;")

        # tagline
        self._lbl_tagline.setText(ex.get("tagline", ""))

        # description
        self._desc.setPlainText(ex.get("description", ""))

        # diagram
        self._diagram.setPlainText(ex.get("diagram", ""))

        # chips
        # clear old chips
        while self._chips_layout.count():
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for nname in ex.get("nodes_used", []):
            chip = QLabel(nname)
            chip.setStyleSheet(
                f"background:transparent; color:{fg_col};"
                f"border:1px solid {fg_col}55;"
                f"border-radius:12px; padding:4px 14px; font-size:11px; font-weight:500;")
            self._chips_layout.addWidget(chip)
        
        self._chips_layout.addStretch() # Ensure they are left-aligned

    # ── load ──────────────────────────────────────────────────────────
    def _on_load(self):
        ex = EXAMPLES[self._selected]
        self._load_cb(ex["graph"])

    # ── theme refresh ─────────────────────────────────────────────────
    def apply_theme(self):
        bg  = Theme.get_str("bg")
        wbg = Theme.get_str("widget_bg")
        txt = Theme.get_str("text")
        brd = Theme.get_str("node_border")

        self.setStyleSheet(f"background:{bg};color:{txt}")
        self.findChild(QWidget, "ex_sidebar").setStyleSheet(
            f"background:{wbg};border-right:2px solid {brd}")
        self._list.setStyleSheet(
            f"QListWidget{{background:transparent;border:none;"
            f"font-family:'Segoe UI';font-size:13px;color:{txt}}}"
            f"QListWidget::item:selected{{background:{brd}55;border-radius:6px}}"
            f"QListWidget::item:hover{{background:{brd}33;border-radius:6px}}")
        self._desc.setStyleSheet(
            f"background:{wbg};color:{txt};border:none;"
            f"font-family:'Cascadia Code';padding:16px")
        self._diagram.setStyleSheet(
            f"background:{wbg};color:{txt};"
            f"border:1px solid {brd};border-radius:8px;padding:12px")
