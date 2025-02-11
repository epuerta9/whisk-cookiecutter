import os
import tempfile
from pathlib import Path
from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import (
    WhiskStorageSchema,
    WhiskStorageResponseSchema,
)
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from kitchenai_llama.storage.llama_parser import Parser
from ..utils.logging import logger
from ..config import settings
from .vector_store import VectorStoreManager

class StorageHandler:
    """Storage Handler for document ingestion and management.
    
    Input Schema (WhiskStorageSchema):
        {
            "id": int,                # Unique document identifier
            "name": str,              # Document name
            "label": str,             # Document label
            "data": bytes | None,     # Document binary data
            "metadata": dict | None,  # Optional metadata
            "extension": str | None   # File extension
        }

    Response Schema (WhiskStorageResponseSchema):
        {
            "id": int,                # Document identifier
            "status": str,            # Status: pending/error/complete/ack
            "error": str | None,      # Error message if any
            "metadata": dict | None,  # Response metadata
            "token_counts": {         # Token usage statistics
                "embedding_tokens": int,
                "llm_prompt_tokens": int,
                "llm_completion_tokens": int,
                "total_llm_tokens": int
            }
        }

    Example:
        >>> with open('document.pdf', 'rb') as f:
        ...     data = f.read()
        >>> storage = WhiskStorageSchema(
        ...     id=1,
        ...     name="document.pdf",
        ...     label="Important Document",
        ...     data=data,
        ...     metadata={"category": "reports"}
        ... )
        >>> response = await handler.handle_storage(storage)
        >>> print(response.status)
        "complete"
    """
    def __init__(self, kitchen: KitchenAIApp):
        self.kitchen = kitchen
        self.vector_store = VectorStoreManager()
        self._register_handlers()

    def _register_handlers(self):
        self.kitchen.storage.handler("storage")(self.handle_storage)
        self.kitchen.storage.on_delete("storage")(self.handle_delete)

    async def handle_storage(self, data: WhiskStorageSchema) -> WhiskStorageResponseSchema:
        """Storage handler for document ingestion"""
        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use the original filename for the temporary file
                temp_file_path = Path(temp_dir) / Path(data.name).name
                
                # Write bytes data to temporary file
                with open(temp_file_path, 'wb') as f:
                    f.write(data.data)
                
                # Initialize parser and load the file
                parser = Parser(api_key=settings.llama_cloud_api_key)
                response = parser.load(str(temp_dir), metadata=data.metadata)
                
                # Setup storage context and process documents
                storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store.store
                )
                
                # Create index with transformations
                VectorStoreIndex.from_documents(
                    response["documents"],
                    storage_context=storage_context,
                    transformations=[
                        TokenTextSplitter(),
                        TitleExtractor(),
                        QuestionsAnsweredExtractor()
                    ],
                    show_progress=True
                )

                return WhiskStorageResponseSchema(
                    id=data.id,
                    name=data.name,
                    label=data.label,
                    data=data.data,
                )
                
        except Exception as e:
            logger.error(f"Error in storage handler: {str(e)}")
            raise

    async def handle_delete(self, data: WhiskStorageSchema) -> None:
        """Storage delete handler"""
        logger.info(f"Deleting storage for {data.id}") 