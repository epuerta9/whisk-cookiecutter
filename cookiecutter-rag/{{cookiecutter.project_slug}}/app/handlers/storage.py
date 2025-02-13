import tempfile
from pathlib import Path
import os
import logging
from whisk.kitchenai_sdk.schema import (
    WhiskStorageSchema,
    WhiskStorageResponseSchema,
    WhiskStorageStatus,
    TokenCountSchema
)
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from kitchenai_llama.storage.llama_parser import Parser

logger = logging.getLogger(__name__)

async def storage_handler(data: WhiskStorageSchema, vector_store=None, token_counter=None) -> WhiskStorageResponseSchema:
    """Storage handler for document ingestion and vectorization.
    
    Args:
        data (WhiskStorageSchema): Storage request with fields:
            - id (int): Unique document ID
            - name (str): Document filename
            - label (str): Handler label (e.g. "storage")
            - data (bytes): Document binary data
            - metadata (dict, optional): Document metadata
            - extension (str, optional): File extension
        vector_store: Vector store for document storage
        token_counter: Counter for tracking token usage
        
    Returns:
        WhiskStorageResponseSchema: Response containing:
            - id (int): Document ID
            - status (WhiskStorageStatus): Processing status
            - error (str, optional): Error message if failed
            - metadata (dict): Document metadata
            - token_counts (TokenCountSchema): Token usage stats
            
    Example:
        >>> with open("doc.pdf", "rb") as f:
        ...     request = WhiskStorageSchema(
        ...         id=1,
        ...         name="doc.pdf",
        ...         label="storage",
        ...         data=f.read(),
        ...         metadata={"category": "technical"}
        ...     )
        >>> response = await storage_handler(request, vector_store)
    """
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use the original filename for the temporary file
            temp_file_path = Path(temp_dir) / Path(data.name).name
            
            # Write bytes data to temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(data.data)
            
            # Initialize parser and load the file
            parser = Parser(api_key=os.environ.get("LLAMA_CLOUD_API_KEY", None))
            response = parser.load(str(temp_dir), metadata=data.metadata)
            
            # Setup storage context and process documents
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
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

            # Get token counts if counter is available
            token_counts = None
            if token_counter:
                token_counts = TokenCountSchema(
                    embedding_tokens=token_counter.total_embedding_token_count,
                    llm_prompt_tokens=token_counter.prompt_llm_token_count,
                    llm_completion_tokens=token_counter.completion_llm_token_count,
                    total_llm_tokens=token_counter.total_llm_token_count
                )
                token_counter.reset_counts()

            # Prepare metadata
            metadata = {
                "document_count": len(response["documents"]),
                "file_name": data.name,
            }
            if data.metadata:
                metadata.update(data.metadata)

            return WhiskStorageResponseSchema(
                id=data.id,
                status=WhiskStorageStatus.COMPLETE,
                metadata=metadata,
                token_counts=token_counts
            )
            
    except Exception as e:
        logger.error(f"Error in storage handler: {str(e)}")
        return WhiskStorageResponseSchema(
            id=data.id,
            status=WhiskStorageStatus.ERROR,
            error=str(e)
        )

async def storage_delete_handler(data: WhiskStorageSchema, vector_store=None) -> None:
    """Handler for deleting documents from storage.
    
    Args:
        data (WhiskStorageSchema): Storage request with fields:
            - id (int): Document ID to delete
            - label (str): Handler label (e.g. "storage")
        vector_store: Vector store to delete from
        
    Returns:
        None
        
    Raises:
        Exception: If deletion fails
            
    Example:
        >>> request = WhiskStorageSchema(
        ...     id=1,
        ...     label="storage"
        ... )
        >>> await storage_delete_handler(request, vector_store)
    """
    try:
        if vector_store and hasattr(vector_store, "delete"):
            # Delete by document ID (convert int to string for ChromaDB)
            await vector_store.adelete(ref_doc_id=str(data.id))
    except Exception as e:
        logger.error(f"Error in storage delete handler: {str(e)}")
        raise 