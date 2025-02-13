from whisk.kitchenai_sdk.schema import (
    WhiskQuerySchema,
    WhiskQueryBaseResponseSchema,
    TokenCountSchema,
    DependencyType
)
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.core.callbacks import TokenCountingHandler
import tiktoken

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

async def query_handler(data: WhiskQuerySchema, llm=None, vector_store=None, system_prompt=None) -> WhiskQueryBaseResponseSchema:
    """Query handler for RAG-based question answering.
    
    Args:
        data (WhiskQuerySchema): Query request with fields:
            - query (str): The question to answer
            - label (str): Handler label (e.g. "query")
            - metadata (dict, optional): Filter metadata (e.g. {"source": "docs"})
            - stream (bool, optional): Enable streaming response
            - stream_id (str, optional): ID for streaming session
        llm: Language model for generating responses
        vector_store: Vector store for document retrieval
        system_prompt (str, optional): System prompt for the LLM
        
    Returns:
        WhiskQueryBaseResponseSchema: Response containing:
            - input (str): Original query
            - output (str): Generated answer
            - retrieval_context (list): Retrieved document chunks
            - metadata (dict): Response metadata
            - token_counts (TokenCountSchema): Token usage stats
            
    Example:
        >>> request = WhiskQuerySchema(
        ...     query="What is RAG?",
        ...     label="query",
        ...     metadata={"source": "technical_docs"}
        ... )
        >>> response = await query_handler(request, llm, vector_store)
    """
    try:
        # Create filters from metadata if provided
        filters = None
        if data.metadata:
            filter_list = [
                MetadataFilter(key=key, value=value)
                for key, value in data.metadata.items()
            ]
            filters = MetadataFilters(filters=filter_list)

        # Create index and query engine with token counter
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            callbacks=[token_counter]
        )
        
        query_engine = index.as_query_engine(
            chat_mode="best",
            filters=filters,
            llm=llm,
            system_prompt=system_prompt,
            verbose=True
        )

        # Execute query
        response = await query_engine.aquery(data.query)

        # Get token counts
        token_counts = TokenCountSchema(
            embedding_tokens=token_counter.total_embedding_token_count,
            llm_prompt_tokens=token_counter.prompt_llm_token_count,
            llm_completion_tokens=token_counter.completion_llm_token_count,
            total_llm_tokens=token_counter.total_llm_token_count
        )
        token_counter.reset_counts()

        # Prepare metadata
        metadata = {"token_counts": token_counts.dict()}
        if data.metadata:
            metadata.update(data.metadata)

        return WhiskQueryBaseResponseSchema.from_llama_response(
            data,
            response,
            metadata=metadata,
            token_counts=token_counts
        )
            
    except Exception as e:
        # Return error response
        return WhiskQueryBaseResponseSchema(
            input=data.query,
            output="Error: " + str(e),
            metadata=data.metadata,
            token_counts=TokenCountSchema()
        ) 