import pytest
from pathlib import Path
import os
from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import DependencyType
from app.dependencies.llm import setup_llm
from app.dependencies.vector_store import setup_vector_store
from app.utils.token_counter import create_token_counter
import tempfile
import shutil

@pytest.fixture
def token_counter():
    """Create token counter for tests"""
    return create_token_counter()

@pytest.fixture
def temp_chroma_dir():
    """Create temporary directory for Chroma DB"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def llm(token_counter):
    """Initialize LLM for tests"""
    return setup_llm(token_counter)

@pytest.fixture
def vector_store(temp_chroma_dir):
    """Initialize vector store for tests"""
    return setup_vector_store(temp_chroma_dir)

@pytest.fixture
def kitchen(llm, vector_store):
    """Initialize KitchenAI app with test dependencies"""
    app = KitchenAIApp(namespace="test")
    
    # Register dependencies
    app.register_dependency(DependencyType.LLM, llm)
    app.register_dependency(DependencyType.VECTOR_STORE, vector_store)
    app.register_dependency(DependencyType.SYSTEM_PROMPT, "You are a helpful assistant.")
    
    return app

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "This is a test document for RAG testing."

@pytest.fixture
def sample_pdf():
    """Sample PDF bytes for testing"""
    return b"%PDF-1.4\n%Test PDF content"

@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {"source": "test", "type": "document"} 