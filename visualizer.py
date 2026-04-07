import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from langchain_community.vectorstores import FAISS

class Visualizer:
    def __init__(self, output_dir: str = "diagrams", db_path: str = "vectordb"):
        self.output_dir = output_dir
        self.db_path = db_path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def draw_architecture(self) -> str:
        """Draws the RAG pipeline architecture diagram."""
        out_path = os.path.join(self.output_dir, "architecture.png")
        
        G = nx.DiGraph()
        nodes = [
            "PDF Upload", "Text Extraction", "Chunking", 
            "Embeddings (all-MiniLM-L6-v2)", "Vector DB (FAISS)", 
            "Similarity Search", "Top-K Chunks", "LLM (Ollama)", "Generated Answer"
        ]
        
        # Add edges sequentially
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i+1])
            
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, scale=2)
        # To make it hierarchical, we map manual positions
        pos = {
            "PDF Upload": (0, 8),
            "Text Extraction": (0, 7),
            "Chunking": (0, 6),
            "Embeddings (all-MiniLM-L6-v2)": (0, 5),
            "Vector DB (FAISS)": (0, 4),
            "Similarity Search": (0, 3),
            "Top-K Chunks": (-1, 2),
            "LLM (Ollama)": (1, 2),
            "Generated Answer": (1, 1)
        }
        # Add a cross edge from Top-K to LLM
        G.add_edge("Top-K Chunks", "LLM (Ollama)")
        
        nx.draw(
            G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_weight='bold', edge_color='gray', 
            arrows=True, arrowsize=20, font_size=9, marker='s'
        )
        
        plt.title("RAG Pipeline Architecture", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, format="png", dpi=150)
        plt.close()
        return out_path

    def draw_vector_space(self) -> str:
        """Projects vectors from FAISS using PCA into 2D space."""
        out_path = os.path.join(self.output_dir, "vector_space.png")
        index_path = os.path.join(self.db_path, "index.faiss")
        
        if not os.path.exists(index_path):
            return self._draw_placeholder(out_path, "No vector database found.")
            
        try:
            # Load basic faiss properties (without full langchain overhead)
            import faiss
            index = faiss.read_index(index_path)
            num_vectors = index.ntotal
            if num_vectors < 2:
                return self._draw_placeholder(out_path, "Not enough vectors for PCA.")
                
            # Reconstruct all vectors
            vectors = np.array([index.reconstruct(i) for i in range(num_vectors)])
            
            # Use PCA
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(vectors)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, color='coral', edgecolors='red')
            plt.title(f"2D PCA Projection of Vector Space ({num_vectors} chunks)", fontsize=14)
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(out_path, format="png", dpi=150)
            plt.close()
            return out_path
            
        except Exception as e:
            return self._draw_placeholder(out_path, f"Error generating PCA: {e}")
            
    def _draw_placeholder(self, out_path: str, message: str) -> str:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, message, horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(out_path, format="png", dpi=150)
        plt.close()
        return out_path

    def draw_data_storage(self) -> str:
        """Draws mapping diagram of chunks to vectors."""
        out_path = os.path.join(self.output_dir, "data_storage.png")
        G = nx.Graph()
        
        G.add_node("Original Document", color="lightgreen", layer=1)
        
        for i in range(3):
            chunk = f"Chunk {i+1}"
            vec = f"[0.1, 0.{i}..]"
            G.add_node(chunk, color="lightblue", layer=2)
            G.add_node(vec, color="salmon", layer=3)
            
            G.add_edge("Original Document", chunk)
            G.add_edge(chunk, vec)
            G.add_edge(vec, "FAISS Database")
            
        G.add_node("FAISS Database", color="orange", layer=4)
        
        pos = nx.multipartite_layout(G, subset_key="layer")
        colors = [nx.get_node_attributes(G, 'color').get(node, 'gray') for node in G.nodes()]
        
        plt.figure(figsize=(9, 5))
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2500, font_size=10, font_weight="bold", edge_color='gray')
        plt.title("Data Storage Workflow", fontsize=14)
        plt.savefig(out_path, format="png", dpi=150)
        plt.close()
        return out_path
