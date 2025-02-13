import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from pathlib import Path

def setup_vector_store(chroma_path: str | Path):
    """Initialize and configure vector store"""
    # Convert Path to string if needed
    chroma_path_str = str(chroma_path) if isinstance(chroma_path, Path) else chroma_path
    
    # Create directory if it doesn't exist
    Path(chroma_path_str).mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB with string path
    chroma_client = chromadb.PersistentClient(path=chroma_path_str)
    chroma_collection = chroma_client.get_or_create_collection("default")
    return ChromaVectorStore(chroma_collection=chroma_collection) 