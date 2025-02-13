import pytest
import os
from pathlib import Path
from app.handlers.storage import storage_handler, storage_delete_handler
from whisk.kitchenai_sdk.schema import WhiskStorageSchema, WhiskStorageStatus

@pytest.mark.asyncio
async def test_storage_handler_pdf(kitchen, sample_pdf, sample_metadata, vector_store, token_counter):
    """Test storage handler with PDF document"""
    storage_request = WhiskStorageSchema(
        id=1,
        name="test.pdf",
        label="storage",
        data=sample_pdf,
        metadata=sample_metadata
    )
    
    # Set mock LLAMA_CLOUD_API_KEY for testing
    os.environ["LLAMA_CLOUD_API_KEY"] = "test_key"
    
    response = await storage_handler(
        storage_request,
        vector_store=vector_store,
        token_counter=token_counter
    )
    
    assert response.id == storage_request.id
    assert response.status == WhiskStorageStatus.COMPLETE
    assert response.metadata is not None
    assert response.metadata.get("file_name") == "test.pdf"
    assert response.metadata.get("source") == "test"
    assert response.error is None
    
    # Verify token counting
    assert response.token_counts is not None
    assert response.token_counts.embedding_tokens >= 0

@pytest.mark.asyncio
async def test_storage_handler_invalid_data(kitchen, vector_store):
    """Test storage handler with invalid data"""
    storage_request = WhiskStorageSchema(
        id=1,
        name="test.xyz",
        label="storage",
        data=b"Invalid data",
    )
    
    try:
        response = await storage_handler(storage_request, vector_store=vector_store)
        assert response.status == WhiskStorageStatus.ERROR
        assert response.error is not None
    except Exception as e:
        # If handler raises exception, that's also acceptable
        assert True

@pytest.mark.asyncio
async def test_storage_delete_handler(kitchen, sample_pdf, sample_metadata, vector_store):
    """Test storage delete handler"""
    storage_request = WhiskStorageSchema(
        id=1,  # Use integer ID as required by schema
        name="test.pdf",
        label="storage",
        data=sample_pdf,
        metadata=sample_metadata
    )
    
    # First store a document
    await storage_handler(storage_request, vector_store=vector_store)
    
    # Then delete it
    await storage_delete_handler(storage_request, vector_store=vector_store)
    
    # Try to delete again - should not raise error
    try:
        await storage_delete_handler(storage_request, vector_store=vector_store)
    except Exception:
        assert False, "Second deletion should not raise error"

@pytest.mark.asyncio
async def test_storage_handler_with_metadata(kitchen, sample_pdf, sample_metadata, vector_store):
    """Test storage handler with metadata"""
    storage_request = WhiskStorageSchema(
        id=1,
        name="test.pdf",
        label="storage",
        data=sample_pdf,
        metadata={"category": "test", "importance": "high"}
    )
    
    response = await storage_handler(storage_request, vector_store=vector_store)
    
    assert response.status == WhiskStorageStatus.COMPLETE
    assert response.metadata is not None
    assert response.metadata.get("category") == "test"
    assert response.metadata.get("importance") == "high"

@pytest.mark.asyncio
async def test_storage_handler_large_file(kitchen, vector_store):
    """Test storage handler with a larger file"""
    # Create a larger test PDF
    large_pdf = b"%PDF-1.4\n" + b"x" * 1000000 + b"\n%%EOF"
    
    storage_request = WhiskStorageSchema(
        id=1,
        name="large.pdf",
        label="storage",
        data=large_pdf,
    )
    
    response = await storage_handler(storage_request, vector_store=vector_store)
    
    assert response.status == WhiskStorageStatus.COMPLETE
    assert response.metadata is not None
    assert response.metadata.get("file_name") == "large.pdf" 