import pytest
from app.handlers.query import query_handler
from whisk.kitchenai_sdk.schema import WhiskQuerySchema, DependencyType

@pytest.mark.asyncio
async def test_query_handler_basic(kitchen, vector_store):
    """Test basic query without metadata"""
    query = WhiskQuerySchema(
        query="What is the meaning of life?",
        label="query"
    )
    
    response = await query_handler(
        query,
        vector_store=vector_store,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.input == query.query
    assert response.output is not None
    assert response.token_counts is not None

@pytest.mark.asyncio
async def test_query_handler_with_metadata(kitchen, vector_store):
    """Test query with metadata filters"""
    query = WhiskQuerySchema(
        query="What is the meaning of life?",
        label="query",
        metadata={"source": "test"}
    )
    
    response = await query_handler(
        query,
        vector_store=vector_store,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.input == query.query
    assert response.output is not None
    assert response.metadata is not None
    assert "token_counts" in response.metadata
    assert response.metadata.get("source") == "test"

@pytest.mark.asyncio
async def test_query_handler_token_counting(kitchen, vector_store, token_counter):
    """Test token counting in query handler"""
    query = WhiskQuerySchema(
        query="What is the meaning of life?",
        label="query"
    )
    
    response = await query_handler(
        query,
        vector_store=vector_store,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.token_counts is not None
    assert response.token_counts.llm_prompt_tokens > 0
    assert response.token_counts.llm_completion_tokens > 0
    assert response.token_counts.total_llm_tokens > 0 