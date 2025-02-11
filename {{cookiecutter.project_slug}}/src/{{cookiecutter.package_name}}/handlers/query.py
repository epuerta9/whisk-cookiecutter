from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import (
    WhiskQuerySchema,
    WhiskQueryBaseResponseSchema,
    TokenCountSchema
)
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from ..utils.logging import logger
from ..config import settings
from .vector_store import get_vector_store, get_llm, token_counter, VectorStoreManager

class QueryHandler:
    """Query Handler for RAG-based question answering.
    
    Input Schema (WhiskQuerySchema):
        {
            "query": str,              # The query text
            "stream": bool = False,    # Whether to stream the response
            "stream_id": str | None,   # Optional stream identifier
            "metadata": dict | None,   # Optional metadata key-value pairs
            "label": str | None,       # Optional label
            "messages": list | None    # Optional chat messages for context
        }

    Response Schema (WhiskQueryBaseResponseSchema):
        {
            "input": str | None,       # Original query
            "output": str | None,      # Generated response
            "retrieval_context": [     # List of relevant source documents
                {
                    "text": str,       # Document text
                    "metadata": dict,  # Document metadata
                    "score": float     # Relevance score
                }
            ],
            "metadata": dict | None,   # Response metadata
            "token_counts": {          # Token usage statistics
                "embedding_tokens": int,
                "llm_prompt_tokens": int,
                "llm_completion_tokens": int,
                "total_llm_tokens": int
            }
        }

    Example:
        >>> query = WhiskQuerySchema(
        ...     query="What is the capital of France?",
        ...     metadata={"source": "geography"}
        ... )
        >>> response = await handler.handle_query(query)
        >>> print(response.output)
        "The capital of France is Paris."
    """
    def __init__(self, kitchen: KitchenAIApp):
        self.kitchen = kitchen
        self.vector_store = VectorStoreManager()
        self._register_handlers()

    def _register_handlers(self):
        self.kitchen.query.handler("query")(self.handle_query)

    async def handle_query(self, data: WhiskQuerySchema) -> WhiskQueryBaseResponseSchema:
        """Query handler with RAG"""
        # Create filters from metadata if provided
        filters = None
        if data.metadata:
            filter_list = [
                MetadataFilter(key=key, value=value)
                for key, value in data.metadata.items()
            ]
            filters = MetadataFilters(filters=filter_list)

        # Create index and query engine
        index = VectorStoreIndex.from_vector_store(self.vector_store.store)
        query_engine = index.as_query_engine(
            chat_mode="best",
            filters=filters,
            llm=self.vector_store.llm,
            verbose=True
        )

        # Execute query
        response = await query_engine.aquery(data.query)

        # Get token counts
        token_counts = self.vector_store.get_token_counts()

        return WhiskQueryBaseResponseSchema.from_llama_response(
            data,
            response,
            token_counts=TokenCountSchema(**token_counts),
            metadata={"token_counts": token_counts, **data.metadata} if data.metadata else {"token_counts": token_counts}
        ) 