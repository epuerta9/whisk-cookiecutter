from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import (
    WhiskEmbedSchema,
    WhiskEmbedResponseSchema,
    TokenCountSchema
)
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from ..utils.logging import logger
from .vector_store import VectorStoreManager

class EmbedHandler:
    """Embedding Handler for text embedding generation and storage.
    
    Input Schema (WhiskEmbedSchema):
        {
            "label": str,             # Embedding label
            "text": str | None,       # Text to embed
            "metadata": dict | None   # Optional metadata
        }

    Response Schema (WhiskEmbedResponseSchema):
        {
            "metadata": dict | None,  # Response metadata
            "token_counts": {         # Token usage statistics
                "embedding_tokens": int,
                "llm_prompt_tokens": int,
                "llm_completion_tokens": int,
                "total_llm_tokens": int
            }
        }

    Example:
        >>> embed_data = WhiskEmbedSchema(
        ...     label="test-embed",
        ...     text="Paris is the capital of France.",
        ...     metadata={"category": "geography"}
        ... )
        >>> response = await handler.handle_embed(embed_data)
        >>> print(response.token_counts.embedding_tokens)
        8
    """
    def __init__(self, kitchen: KitchenAIApp):
        self.kitchen = kitchen
        self.vector_store = VectorStoreManager()
        self._register_handlers()

    def _register_handlers(self):
        self.kitchen.embeddings.handler("embed")(self.handle_embed)

    async def handle_embed(self, data: WhiskEmbedSchema) -> WhiskEmbedResponseSchema:
        """Embedding handler"""
        try:
            # Create document and index it
            document = Document(text=data.text, metadata=data.metadata)
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store.store
            )
            
            VectorStoreIndex.from_documents(
                [document],
                storage_context=storage_context,
                transformations=[
                    TokenTextSplitter(),
                    TitleExtractor(),
                    QuestionsAnsweredExtractor()
                ],
                show_progress=True
            )

            token_counts = self.vector_store.get_token_counts()

            return WhiskEmbedResponseSchema(
                text=data.text,
                token_counts=TokenCountSchema(**token_counts),
                metadata={"token_counts": token_counts, **data.metadata} if data.metadata else {"token_counts": token_counts}
            )
        except Exception as e:
            logger.error(f"Error in embed handler: {str(e)}")
            raise 