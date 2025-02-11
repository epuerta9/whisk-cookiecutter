import pytest
from {{cookiecutter.package_name}}.handlers.vector_store import get_vector_store, get_llm

def test_vector_store_initialization(temp_chroma_dir):
    # Test that vector store can be initialized
    try:
        vector_store = get_vector_store()
        assert vector_store is not None
    except Exception as e:
        pytest.fail(f"Vector store initialization failed: {e}")

def test_llm_initialization():
    # Test that LLM can be initialized
    try:
        llm = get_llm()
        assert llm is not None
    except Exception as e:
        pytest.fail(f"LLM initialization failed: {e}") 