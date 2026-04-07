import time
import os
from typing import Dict, Any, Tuple, List
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils import estimate_tokens

class RAGEngine:
    def __init__(self, db_path: str = "vectordb", model_name: str = "llama3.2:latest"):
        self.db_path = db_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = Ollama(model=self.model_name)
        self.vector_db = None
        self.retrieval_chain = None
        
        # System prompt instructions
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise.\n\n"
            "{context}"
        )
        
    def load_db(self) -> bool:
        """Loads FAISS index from disk. Returns True if successful."""
        if os.path.exists(os.path.join(self.db_path, "index.faiss")):
            self.vector_db = FAISS.load_local(
                self.db_path, 
                self.embeddings,
                allow_dangerous_deserialization=True # Necessary for local FAISS usage
            )
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{input}"),
            ])
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
                
            setup_and_retrieval = RunnableParallel(
                context=retriever,
                input=RunnablePassthrough()
            )
            
            self.retrieval_chain = setup_and_retrieval.assign(
                answer=(
                    RunnablePassthrough.assign(
                        context=lambda x: format_docs(x["context"])
                    )
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
            )
            return True
        return False
        
    def clear_db(self) -> bool:
        """Deletes the FAISS index from disk and resets state."""
        import shutil
        self.vector_db = None
        self.retrieval_chain = None
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            return True
        return False

    def retrieve(self, question: str, k: int = 4):
        """Retrieve similar chunks from FAISS. Returns list of Documents."""
        if not self.vector_db:
            return []
        return self.vector_db.similarity_search(question, k=k)

    def generate(self, prompt_text: str) -> str:
        """Send a fully-formed prompt string to the LLM. Returns answer string."""
        return self.llm.invoke(prompt_text)

    def query(self, question: str) -> Dict[str, Any]:
        """Runs the query through the RAG pipeline."""
        if not self.vector_db:
            return {"error": "Database not loaded. Please index documents first."}

        start_time = time.time()
        docs = self.retrieve(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        full_prompt = (
            f"{self.system_prompt.replace('{context}', context)}\n\n"
            f"Question: {question}"
        )
        answer = self.generate(full_prompt)
        latency = time.time() - start_time

        tokens_sent = estimate_tokens(question) + estimate_tokens(context) + estimate_tokens(self.system_prompt)

        return {
            "answer": answer,
            "sources": docs,
            "latency": round(latency, 2),
            "tokens_sent": tokens_sent
        }
