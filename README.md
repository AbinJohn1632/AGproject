# Visual RAG Node Orchestrator

A powerful, Figma-inspired node-based IDE for building, testing, and orchestrating Local Retrieval-Augmented Generation (RAG) pipelines. This system allows users to visually design complex AI workflows using a drag-and-drop interface, leveraging local LLMs (via Ollama) and vector databases (FAISS).

## 🚀 Key Features

- **Visual Graph Editor**: Drag-and-drop interface for connecting AI components.
- **Local Execution**: Complete privacy using Ollama and local vector stores.
- **State Management**: Built-in nodes for persistent conversation memory and semantic caching.
- **Real-time Tracing**: Visual feedback of data flow through the graph during execution.
- **Topological Sorting**: Automatic execution order based on graph connections.

---

## 🧩 Node Reference

The orchestrator includes over 20 specialized nodes to build sophisticated RAG pipelines:

### 📥 Input & Output
- **Query Input**: The entry point for user questions.
- **Response Output**: Displays the final generated answer from the LLM.

### 🧠 LLM & AI
- **Ollama LLM**: Connects to local models (e.g., `llama3.2`). Handles generation based on the provided prompt.

### 📂 Vector DB & Retrieval
- **PDF Loader**: Parses PDF documents into searchable text chunks.
- **FAISS DB**: The core vector database. Indices documents and retrieves relevant context based on query similarity.
- **Top-K Retriever**: A specialized wrapper for direct FAISS similarity search with configurable `k`.
- **Reranker**: Sorts retrieved chunks based on relevance scores and keeps only the top-K.
- **Score Filter**: Filters out retrieved chunks that fall below a specific relevance threshold.

### 💾 Memory & Cache
- **Buffer**: Stores conversation history to provide the LLM with context from previous turns.
- **Cache**: A semantic cache that stores query-answer pairs. It "short-circuits" the graph on a hit, providing instant responses.
- **Conversation Starter**: Injects initial system context or greetings before the first user query.
- **Seed Buffer Loader**: Allows pre-loading the conversation buffer with specific history turns.
- **Memory Formatter**: Condenses or truncates long conversation histories to fit within LLM token limits.

### ⚙️ Processing & Logic
- **Prompt Template**: Formats raw data (query, context, memory) into a structured string using placeholders like `{query}`.
- **System Message**: Emits constant instructions (e.g., "You are a helpful assistant") to the model.
- **Merge**: Combines two inputs into one.
- **Multi Merge**: Merges up to 4 inputs (System + Memory + Context + Query).
- **Copy**: Duplicates data to branch out the pipeline into parallel paths.
- **Router**: Directs data to different paths based on keywords or confidence scores.

### 🔍 Debugging
- **Debug Inspector**: A "pass-through" node that displays the data currently moving through that specific edge.

---

## 🛠 Setup & Installation

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: Download from [ollama.com](https://ollama.com/) and pull a model:
  ```bash
  ollama pull llama3.2
  ```

### 2. Installation
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage
Run the main application:
```bash
python gui.py
```

---

## 📂 Project Structure

- `gui.py`: Main application entry point and UI logic.
- `node_editor.py`: Visual graph editor component and node definitions.
- `graph_executor.py`: The engine that traverses and executes the node graph.
- `rag_engine.py`: Core RAG logic (Ollama integration, FAISS management).
- `ingest.py`: PDF processing and embedding logic.
- `tests/`: Comprehensive test suite for graph execution.
- `vectordb/`: Default storage location for persistent FAISS indices.

---

