import pytest
from {{cookiecutter.package_name}}.handlers.embed import setup_embed_handler

{% if cookiecutter.include_embed_handler == "y" %}
@pytest.mark.asyncio
async def test_embed_handler_setup(kitchen):
    # Test that the handler can be set up without errors
    setup_embed_handler(kitchen)
    assert "embed" in kitchen.embeddings.handlers

@pytest.mark.asyncio
async def test_embed_handler_text_embedding(kitchen, sample_embed_data):
    # Set up the handler
    setup_embed_handler(kitchen)
    handler = kitchen.embeddings.handlers["embed"]
    
    # Test text embedding
    response = await handler(sample_embed_data)
    
    # Verify response
    assert response.text == sample_embed_data.text
    assert response.token_counts is not None
    assert "token_counts" in response.metadata

@pytest.mark.asyncio
async def test_embed_handler_with_metadata(kitchen):
    # Set up the handler
    setup_embed_handler(kitchen)
    handler = kitchen.embeddings.handlers["embed"]
    
    # Create embed request with metadata
    embed_data = WhiskEmbedSchema(
        text="Test embedding text",
        metadata={"source": "test", "category": "general"}
    )
    
    # Test embedding with metadata
    response = await handler(embed_data)
    
    # Verify metadata handling
    assert response.metadata["source"] == "test"
    assert response.metadata["category"] == "general"
{% endif %} 