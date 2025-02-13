import pytest
from app.handlers.memory import memory_handler, clear_memory_handler, memory_manager
from whisk.kitchenai_sdk.schema import WhiskQuerySchema, DependencyType

@pytest.mark.asyncio
async def test_memory_handler_basic(kitchen):
    """Test basic memory chat"""
    query = WhiskQuerySchema(
        query="Hello!",
        label="memory"
    )
    
    response = await memory_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.input == query.query
    assert response.output is not None
    assert response.messages is not None
    assert len(response.messages) >= 2

@pytest.mark.asyncio
async def test_memory_persistence(kitchen):
    """Test that memory persists between calls"""
    # First message
    query1 = WhiskQuerySchema(
        query="What is your name?",
        label="memory"
    )
    
    response1 = await memory_handler(
        query1,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    # Second message referencing first
    query2 = WhiskQuerySchema(
        query="What did I just ask you?",
        label="memory"
    )
    
    response2 = await memory_handler(
        query2,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert len(response2.messages) > len(response1.messages)
    assert "name" in response2.output.lower()

@pytest.mark.asyncio
async def test_memory_clear(kitchen):
    """Test memory clearing"""
    # Add a message
    query = WhiskQuerySchema(
        query="Remember this message",
        label="memory"
    )
    
    await memory_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    # Clear memory
    clear_query = WhiskQuerySchema(
        query="clear",
        label="clear_memory"
    )
    
    response = await clear_memory_handler(clear_query)
    
    assert response.output == "Memory cleared successfully"
    assert len(memory_manager.get_history()) == 0

@pytest.mark.asyncio
async def test_memory_token_counting(kitchen):
    """Test token counting in memory chat"""
    query = WhiskQuerySchema(
        query="Count my tokens",
        label="memory"
    )
    
    response = await memory_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.token_counts is not None
    assert response.token_counts.llm_prompt_tokens > 0
    assert response.token_counts.llm_completion_tokens > 0
    assert response.token_counts.total_llm_tokens > 0 