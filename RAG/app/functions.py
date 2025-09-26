import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

import warnings
warnings.filterwarnings('ignore')

class RAGSystem:    
    def __init__(self, ollama_base="http://localhost:11434"):
        self.ollama_base = ollama_base
        self.vector_db = None
        self.embedding_model = None
        self.chat_model = None
        
    def setup_embedding_model(self, model_name="nomic-embed-text"):
        try:
            self.embedding_model = OllamaEmbeddings(
                model=model_name,
                base_url=self.ollama_base,
                show_progress=False
            )
            return True, "Embedding model initialized."
        except Exception as e:
            return False, f"Failed to initialize embedding model: {str(e)}"
    
    def setup_chat_model(self, model_name="llama3.2", temperature=0.2):
        try:
            self.chat_model = ChatOllama(
                model=model_name,
                base_url=self.ollama_base,
                temperature=temperature,
                top_p=0.9,
                num_predict=512
            )
            return True, "Chat model initialized."
        except Exception as e:
            return False, f"Failed to initialize chat model: {str(e)}"

    
    def process_pdf(self, file_path, chunk_size=1000, chunk_overlap=200):
        try:
            if not self.embedding_model:
                return False, "Please set up the embedding model first.", 0, 0

            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(pages)
            

            self.vector_db = Chroma.from_documents(
                chunks,
                self.embedding_model,
                persist_directory="./streamlit_chroma_db"
            )
            
            return True, "Document processed successfully.", len(pages), len(chunks)

        except Exception as e:
            return False, f"Failed to process document: {str(e)}", 0, 0
    
    def _get_document_prompt(self, docs):
        prompt = "\n"
        for doc in docs:
            prompt += "\nContent:\n"
            prompt += doc.page_content + "\n\n"
        return prompt
    
    def ask_question(self, question, k=5):
        try:
            if not self.vector_db:
                return "Please upload and process a PDF document first.", []
            
            if not self.chat_model:
                return "Please set up the chat model first.", []
            
            retrieved_docs = self.vector_db.similarity_search(question, k=k)
            
            formatted_context = self._get_document_prompt(retrieved_docs)
            
            prompt = f"""
## SYSTEM
You are a knowledgeable and factual assistant.
Answer **only** using the provided CONTEXT. If the answer cannot be found in the context,
reply exactly: "The provided context does not contain this information."

## USER QUESTION
{question}

## CONTEXT
{formatted_context}

## REQUIREMENTS
- Be concise and clear
- Include **Source** with file name and page numbers used.
- No speculation
- Answer in the same language as the question
"""
            
            response = self.chat_model.invoke(prompt)
            return response.content.strip(), retrieved_docs
            
        except Exception as e:
            return f"Something went wrong: {str(e)}", []
    
    def get_system_status(self):
        status = {
            "embedding_ready": self.embedding_model is not None,
            "chat_ready": self.chat_model is not None,
            "vector_db_ready": self.vector_db is not None,
            "fully_ready": all([
                self.embedding_model is not None,
                self.chat_model is not None,
                self.vector_db is not None
            ])
        }
        return status

def create_rag_system(ollama_base="http://localhost:11434"):
    return RAGSystem(ollama_base)

def quick_setup(rag_system, embedding_model="nomic-embed-text", chat_model="llama3.2"):
    
    embed_success, embed_msg = rag_system.setup_embedding_model(embedding_model)
    if not embed_success:
        return False, embed_msg
    
    chat_success, chat_msg = rag_system.setup_chat_model(chat_model)
    if not chat_success:
        return False, chat_msg
    
    return True, "RAG System setup successfully completed..."