import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class ComplianceRAG:
    def __init__(self, index_dir: str = "data/faiss_index", embedding_model: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        # Load local sentence-transformers embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Load FAISS index if it exists on disk, otherwise set to None
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            self.db = FAISS.load_local(
                index_dir, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            self.db = None
            
    def ingest_document(self, text: str, doc_name: str = "Document") -> int:
        if not text or not text.strip():
            return 0
            
        # Split document into text chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_text(text)
        
        if not chunks:
            return 0
            
        docs = [Document(page_content=chunk, metadata={"source": doc_name}) for chunk in chunks]
        
        # Add to vector store
        if self.db is None:
            self.db = FAISS.from_documents(docs, self.embeddings)
        else:
            self.db.add_documents(docs)
            
        # Save index to disk
        os.makedirs(self.index_dir, exist_ok=True)
        self.db.save_local(self.index_dir)
        
        return len(chunks)
        
    def query(self, question: str, groq_api_key: str = None) -> dict:
        if self.db is None:
            return {
                "answer": "No documents ingested in the RAG corpus yet. Please upload files to search.",
                "retrieved_chunks": [],
                "mode": "fallback"
            }
            
        # Retrieve top 3 relevant passages
        results = self.db.similarity_search(question, k=3)
        retrieved_chunks = [doc.page_content for doc in results]
        
        if not retrieved_chunks:
            return {
                "answer": "No relevant passages found in the corpus for your query.",
                "retrieved_chunks": [],
                "mode": "fallback"
            }
            
        # Load Groq API Key
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        if api_key and api_key.strip():
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                
                # Context aggregation
                context = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}]: {doc.page_content}" for doc in results])
                
                prompt = (
                    "You are a Senior Regulatory Compliance Analyst. Use the following retrieved "
                    "legal/compliance context passages to answer the user's question. Ground your answer "
                    "strictly in the context provided. If the answer cannot be found in the context, say "
                    "so explicitly — do not hallucinate or make up facts.\n\n"
                    f"--- Retrieved Context Passages ---\n{context}\n\n"
                    f"Question: {question}\nAnswer:"
                )
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a professional regulatory compliance Q&A assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.1
                )
                
                answer = chat_completion.choices[0].message.content
                return {
                    "answer": answer,
                    "retrieved_chunks": retrieved_chunks,
                    "mode": "online"
                }
            except Exception as e:
                # API Call error fallback
                fallback_answer = self._generate_local_fallback_report(question, results, error=str(e))
                return {
                    "answer": fallback_answer,
                    "retrieved_chunks": retrieved_chunks,
                    "mode": "fallback"
                }
        else:
            # Offline local fallback
            fallback_answer = self._generate_local_fallback_report(question, results)
            return {
                "answer": fallback_answer,
                "retrieved_chunks": retrieved_chunks,
                "mode": "fallback"
            }
            
    def _generate_local_fallback_report(self, question: str, docs, error: str = None) -> str:
        report = []
        report.append("# RAG Compliance Summary Report (Local Offline Synthesizer)")
        if error:
            report.append(f"> [!WARNING]\n> Groq LLM API call failed: `{error}`. Automatically falling back to local context synthesis report.\n")
        else:
            report.append("> [!NOTE]\n> Groq LLM API Key is absent. Automatically falling back to offline local context synthesis report.\n")
            
        report.append(f"### Question Asked\n*{question}*\n")
        report.append("### Key Matching Compliance Passages Found\n")
        
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "Uploaded Document")
            report.append(f"#### **Passage {i} (Source: {src})**")
            report.append(f"```text\n{doc.page_content.strip()}\n```\n")
            
        report.append("---")
        report.append("*End of Local Offline Synthesis. For interactive LLM conversational synthesis, please configure a valid `GROQ_API_KEY`.*")
        return "\n".join(report)
