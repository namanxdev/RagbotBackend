import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import dotenv
import uuid

dotenv.load_dotenv()

class RAGSystem:
    def __init__(self, pdf_path: str):
        """Initialize RAG system with a PDF file"""
        self.pdf_path = pdf_path
        self.chunks = []
        self.vector_store = None
        self.llm = None
        self.graph = None
        
        # Load environment variables
        self.GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
        self.DATABASE_URL = os.environ["DATABASE_URL"]
        
        # Initialize components
        self._initialize_models()
        self._process_pdf()
        self._setup_vector_store()
        self._build_graph()
    
    def _initialize_models(self):
        """Initialize chat model and embeddings"""
        self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=self.GEMINI_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=self.GEMINI_API_KEY)
    
    def _process_pdf(self):
        """Load and chunk the PDF document"""
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunks = text_splitter.split_documents(documents)
        print(f"Number of chunks created: {len(self.chunks)}")
    
    def _setup_vector_store(self):
        """Create and populate vector store"""
        # Create unique table name for this session
        table_name = f'docs_{str(uuid.uuid4()).replace("-", "_")}'
        
        # Postgres connection initialization
        pg_engine = PGEngine.from_connection_string(url=self.DATABASE_URL)
        
        try:
            # Method 1: Try the newer approach with proper schema
            self.vector_store = PGVectorStore.create_sync(
                embedding_service=self.embeddings,
                engine=pg_engine,
                table_name=table_name,
                auto_create_table=True
            )
            print("Vector store created successfully")
            
            # Add documents to the vector store
            self.vector_store.add_documents(self.chunks)
            print(f"Vector store populated with {len(self.chunks)} documents")
            
        except Exception as e:
            print(f"Error with PGVectorStore: {e}")
            print("Trying alternative approach with PGVector...")
            
            # Method 2: Fallback to alternative implementation
            try:
                from langchain_postgres.vectorstores import PGVector
                
                self.vector_store = PGVector(
                    embeddings=self.embeddings,
                    connection=self.DATABASE_URL,
                    collection_name=table_name
                )
                
                # Add documents
                self.vector_store.add_documents(self.chunks)
                print(f"Alternative vector store created with {len(self.chunks)} documents")
                
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                print("Trying manual table creation...")
                
                
    def _build_graph(self):
        """Build the RAG graph"""
        # Custom prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions about documents. Use the provided context to answer the questions. IMPORTANT: If you don't know the answer, say 'I don't know'. Do not make up answers."),
            ("user", "Question: {question}\nContext: {context}")
        ])
        
        # Define state for application
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
        
        def retrieve(state: State):
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}
        
        def generate(state: State):
            context = "\n\n".join([doc.page_content for doc in state["context"]])
            message = self.prompt.invoke({"question": state["question"], "context": context})
            response = self.llm.invoke(message)
            return {"answer": response.content}
        
        # Compile application
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer"""
        try:
            result = self.graph.invoke({"question": question})
            return result["answer"]
        except Exception as e:
            return f"Error: {e}"
    
    def ask_question_with_sources(self, question: str):
        """Ask a question and get answer with source documents"""
        try:
            # Get relevant documents
            retrieved_docs = self.vector_store.similarity_search(question, k=3)
            
            # Generate answer
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            message = self.prompt.invoke({"question": question, "context": context})
            response = self.llm.invoke(message)
            
            # Extract source snippets
            source_docs = [doc.page_content[:200] + "..." for doc in retrieved_docs]
            
            return response.content, source_docs
        except Exception as e:
            return f"Error: {e}", []
    
    def get_chunks_count(self) -> int:
        """Get the number of chunks created"""
        return len(self.chunks)


# # Legacy code for backward compatibility
# GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# print("Using API key:", GEMINI_API_KEY)

# # Chat model initialization
# llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=GEMINI_API_KEY)

# file_path = "Sample_pdf.pdf"
# loader = PyPDFLoader(file_path)
# documents = loader.load()
# print(documents[6].page_content[:500])

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,separators=["\n\n", "\n", " ", ""])
# all_chunks = text_splitter.split_documents(documents)
# print(f"Number of chunks: {len(all_chunks)}")

# # Embeddings model initialization
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY)

# # Database connection
# DATABASE_URL = os.environ["DATABASE_URL"]

# # Postgres connection initialization
# pg_engine = PGEngine.from_connection_string(url=DATABASE_URL)

# # Create vector store with documents
# try:
#     vector_store = PGVectorStore.create_sync(
#         embedding_service=embeddings,
#         engine=pg_engine,
#         table_name='document_embeddings_clean'
#     )
#     print("Vector store created successfully")
    
#     # Add documents to the vector store
#     vector_store.add_documents(all_chunks)
#     print(f"Vector store populated with {len(all_chunks)} documents")
    
# except Exception as e:
#     print(f"Error creating vector store: {e}")
#     print("Trying alternative approach...")
    
#     # Alternative: Try without pre-creating, let add_documents create it
#     try:
#         from langchain_postgres.vectorstores import PGVector

#         vector_store = PGVector(
#             embeddings=embeddings,
#             connection=DATABASE_URL,
#             collection_name='document_embeddings'
#         )
#         vector_store.add_documents(all_chunks)
#         print(f"Vector store created with {len(all_chunks)} documents")
#     except Exception as e2:
#         print(f"Alternative approach also failed: {e2}")

# # Custom prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that answers questions about documents. Use the provided context to answer the questions. IMPORTANT: If you don't know the answer, say 'I don't know'. Do not make up answers."),
#     ("user", "Question: {question}\nContext: {context}")
# ])

# # Define state for application
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str


# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}

# def generate(state: State):
#     context = "\n\n".join([doc.page_content for doc in state["context"]])
#     message = prompt.invoke({"question": state["question"], "context": context})
#     response = llm.invoke(message)
#     return {"answer": response.content}


# # Compile application and test
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()

# # Function to ask questions
# def ask_question(question: str):
#     """Ask a question to the RAG system"""
#     try:
#         result = graph.invoke({"question": question})
#         return result["answer"]
#     except Exception as e:
#         return f"Error: {e}"

# # Example usage and testing functions (for standalone use)
# if __name__ == "__main__":
#     # Test with sample PDF if it exists
#     sample_pdf = "Sample_pdf.pdf"
#     if os.path.exists(sample_pdf):
#         print("Testing with sample PDF...")
#         rag_system = RAGSystem(sample_pdf)
        
#         # Interactive mode
#         print("\n" + "="*50)
#         print("Interactive Mode - Ask your own questions!")
#         print("Type 'quit' to exit")
#         print("="*50)
        
#         while True:
#             user_question = input("\nYour question: ").strip()
#             if user_question.lower() in ['quit', 'exit', 'q']:
#                 print("Goodbye!")
#                 break
#             if user_question:
#                 answer = rag_system.ask_question(user_question)
#                 print(f"Answer: {answer}")
#             else:
#                 print("Please enter a valid question.")
#     else:
#         print(f"Sample PDF '{sample_pdf}' not found. Please ensure the file exists or use the FastAPI backend.")