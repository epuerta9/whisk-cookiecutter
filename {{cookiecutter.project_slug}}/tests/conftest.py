import pytest
from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import (
    WhiskQuerySchema,
    WhiskStorageSchema,
    WhiskEmbedSchema
)
import chromadb
import tempfile
import os

@pytest.fixture
def kitchen():
    return KitchenAIApp(namespace="test-llama-index")

@pytest.fixture
def temp_chroma_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def chroma_client(temp_chroma_dir):
    client = chromadb.PersistentClient(path=temp_dir)
    yield client
    client.reset()

@pytest.fixture
def sample_query_data():
    return WhiskQuerySchema(
        query="What is the capital of France?",
        metadata={"source": "test"}
    )

@pytest.fixture
def sample_storage_data():
    # Create a small test PDF file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"This is a test document about Paris being the capital of France.")
        filepath = f.name

    with open(filepath, 'rb') as f:
        data = f.read()

    os.unlink(filepath)  # Clean up the temporary file

    return WhiskStorageSchema(
        id="test-doc-1",
        name="test.txt",
        label="Test Document",
        data=data,
        metadata={"source": "test"}
    )

@pytest.fixture
def sample_embed_data():
    return WhiskEmbedSchema(
        text="Paris is the capital of France.",
        metadata={"source": "test"}
    ) 