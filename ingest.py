import os
import time
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class DocIngestor:
    def __init__(self, db_path: str = "vectordb"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

    def ingest_pdfs(self, pdf_paths: List[str], progress_callback=None) -> Dict[str, Any]:
        """
        Ingests a list of PDF file paths, converts them to chunks, generates embeddings,
        saves to a local FAISS database, and returns runtime statistics.
        """
        start_time = time.time()
        
        all_docs = []
        total_pages = 0
        total_chars = 0
        
        # Load PDFs
        for i, path in enumerate(pdf_paths):
            if progress_callback:
                progress_callback(int((i / len(pdf_paths)) * 30), f"Loading {os.path.basename(path)}...")
            
            loader = PyPDFLoader(path)
            docs = loader.load()
            
            for doc in docs:
                total_pages += 1
                total_chars += len(doc.page_content)
                
            all_docs.extend(docs)
            
        if progress_callback:
            progress_callback(40, "Chunking texts...")
            
        # Chunk text
        chunks = self.text_splitter.split_documents(all_docs)
        num_chunks = len(chunks)
        avg_chunk_size = sum(len(c.page_content) for c in chunks) / num_chunks if num_chunks > 0 else 0
        
        if progress_callback:
            progress_callback(60, f"Generating embeddings for {num_chunks} chunks...")
            
        # Embed and Create DB
        vector_db = FAISS.from_documents(chunks, self.embeddings)
        
        if progress_callback:
            progress_callback(90, "Saving local database...")
            
        # Save locally
        vector_db.save_local(self.db_path)
        
        indexing_time = time.time() - start_time
        
        if progress_callback:
            progress_callback(100, "Done!")

        return {
            "num_pdfs": len(pdf_paths),
            "total_pages": total_pages,
            "total_chars": total_chars,
            "num_chunks": num_chunks,
            "avg_chunk_size": int(avg_chunk_size),
            "indexing_time": round(indexing_time, 2),
            "dimension": 384  # MiniLM-L6-v2 dimension
        }
