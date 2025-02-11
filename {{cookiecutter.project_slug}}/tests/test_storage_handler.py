import pytest
from {{cookiecutter.package_name}}.handlers.storage import setup_storage_handler

{% if cookiecutter.include_storage_handler == "y" %}
@pytest.mark.asyncio
async def test_storage_handler_setup(kitchen):
    # Test that the handler can be set up without errors
    setup_storage_handler(kitchen)
    assert "storage" in kitchen.storage.handlers

@pytest.mark.asyncio
async def test_storage_handler_document_ingestion(kitchen, sample_storage_data):
    # Set up the handler
    setup_storage_handler(kitchen)
    handler = kitchen.storage.handlers["storage"]
    
    # Test document ingestion
    response = await handler(sample_storage_data)
    
    # Verify response
    assert response.id == sample_storage_data.id
    assert response.name == sample_storage_data.name
    assert response.label == sample_storage_data.label

@pytest.mark.asyncio
async def test_storage_delete_handler(kitchen, sample_storage_data):
    # Set up the handler
    setup_storage_handler(kitchen)
    delete_handler = kitchen.storage.delete_handlers["storage"]
    
    # Test delete operation
    try:
        await delete_handler(sample_storage_data)
    except Exception as e:
        pytest.fail(f"Delete handler raised an exception: {e}")
{% endif %} 