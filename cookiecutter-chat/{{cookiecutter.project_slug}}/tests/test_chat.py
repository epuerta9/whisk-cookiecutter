import pytest
from app.handlers.chat import chat_handler
from whisk.kitchenai_sdk.schema import WhiskQuerySchema, DependencyType

@pytest.mark.asyncio
async def test_chat_handler_basic(kitchen):
    """Test basic chat without history"""
    query = WhiskQuerySchema(
        query="Hello!",
        label="chat"
    )
    
    response = await chat_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.input == query.query
    assert response.output is not None
    assert response.messages is not None
    assert len(response.messages) >= 2  # System + user + assistant

@pytest.mark.asyncio
async def test_chat_handler_with_history(kitchen):
    """Test chat with message history"""
    history = [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am a Shakespearean assistant."}
    ]
    
    query = WhiskQuerySchema(
        query="Tell me more.",
        label="chat",
        messages=history
    )
    
    response = await chat_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.messages is not None
    assert len(response.messages) > len(history)
    assert response.metadata.get("personality") == "{{ cookiecutter.personality }}"

@pytest.mark.asyncio
async def test_chat_handler_token_counting(kitchen):
    """Test token counting in chat"""
    query = WhiskQuerySchema(
        query="Write a sonnet.",
        label="chat"
    )
    
    response = await chat_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.token_counts is not None
    assert response.token_counts.llm_prompt_tokens > 0
    assert response.token_counts.llm_completion_tokens > 0
    assert response.token_counts.total_llm_tokens > 0 