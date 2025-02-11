import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken
from ..config import settings

{% if cookiecutter.framework == 'llama-index' %}
# LlamaIndex vector store implementation
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
{% endif %}
{% if cookiecutter.framework == 'langchain' %}
# Langchain vector store implementation
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
{% endif %}

class VectorStoreManager:
    """Singleton manager for vector store and LLM resources.
    
    This class manages shared resources including:
    - ChromaDB vector store
    - OpenAI LLM instance
    - Token counting utilities

    Token Count Schema (TokenCountSchema):
        {
            "embedding_tokens": int | None,     # Tokens used for embeddings
            "llm_prompt_tokens": int | None,    # Tokens in prompts
            "llm_completion_tokens": int | None, # Tokens in completions
            "total_llm_tokens": int | None      # Total tokens used
        }

    Example:
        >>> manager = VectorStoreManager()
        >>> vector_store = manager.store
        >>> llm = manager.llm
        >>> token_counts = manager.get_token_counts()
        >>> print(token_counts["total_llm_tokens"])
        42
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the vector store and LLM components"""
        # Initialize token counter
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
        )
        Settings.callback_manager = CallbackManager([self.token_counter])

        # Initialize LLM
        self.llm = OpenAI(model="gpt-3.5-turbo")
        Settings.llm = self.llm

        # Initialize vector store
        {% if cookiecutter.use_persistent_chroma == "y" %}
        chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        {% else %}
        chroma_client = chromadb.Client()
        {% endif %}
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        self.store = ChromaVectorStore(chroma_collection=chroma_collection)

    def get_token_counts(self):
        """Get current token counts and reset counter"""
        counts = {
            "embedding_tokens": self.token_counter.total_embedding_token_count,
            "llm_prompt_tokens": self.token_counter.prompt_llm_token_count,
            "llm_completion_tokens": self.token_counter.completion_llm_token_count,
            "total_llm_tokens": self.token_counter.total_llm_token_count
        }
        self.token_counter.reset_counts()
        return counts 