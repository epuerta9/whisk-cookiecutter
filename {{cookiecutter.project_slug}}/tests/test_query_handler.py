import pytest
from {{cookiecutter.package_name}}.handlers.query import setup_query_handler
from {{cookiecutter.package_name}}.handlers.vector_store import get_vector_store, get_llm

{% if cookiecutter.include_query_handler == "y" %}
@pytest.mark.asyncio
async def test_query_handler_setup(kitchen):
    # Test that the handler can be set up without errors
    setup_query_handler(kitchen)
    assert "query" in kitchen.query.handlers

@pytest.mark.asyncio
async def test_query_handler_response(kitchen, sample_query_data):
    # Set up the handler
    setup_query_handler(kitchen)
    handler = kitchen.query.handlers["query"]
    
    # Execute the query
    response = await handler(sample_query_data)
    
    # Verify response structure
    assert response.query == sample_query_data.query
    assert response.token_counts is not None
    assert "token_counts" in response.metadata

@pytest.mark.asyncio
async def test_query_handler_with_metadata_filters(kitchen):
    # Set up the handler
    setup_query_handler(kitchen)
    handler = kitchen.query.handlers["query"]
    
    # Create query with metadata filters
    query_data = WhiskQuerySchema(
        query="What is the capital of France?",
        metadata={"source": "test", "category": "geography"}
    )
    
    # Execute the query
    response = await handler(query_data)
    
    # Verify metadata handling
    assert response.metadata["source"] == "test"
    assert response.metadata["category"] == "geography"
{% endif %} 