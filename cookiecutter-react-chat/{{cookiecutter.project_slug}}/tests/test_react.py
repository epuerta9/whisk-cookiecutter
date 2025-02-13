import pytest
from app.handlers.react import react_handler, TOOLS
from whisk.kitchenai_sdk.schema import WhiskQuerySchema, DependencyType

@pytest.mark.asyncio
async def test_react_handler_basic(kitchen):
    """Test basic ReAct chat without tool use"""
    query = WhiskQuerySchema(
        query="Hello!",
        label="react"
    )
    
    response = await react_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.input == query.query
    assert response.output is not None
    assert response.messages is not None
    assert len(response.messages) >= 2

@pytest.mark.asyncio
async def test_react_handler_with_tool(kitchen):
    """Test ReAct chat with tool usage"""
    query = WhiskQuerySchema(
        query="What's 2 + 2?",
        label="react"
    )
    
    response = await react_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.metadata is not None
    assert "tool_usage" in response.metadata
    assert len(response.metadata["tool_usage"]) > 0

@pytest.mark.asyncio
async def test_react_handler_token_counting(kitchen):
    """Test token counting in ReAct chat"""
    query = WhiskQuerySchema(
        query="Search for information about Python.",
        label="react"
    )
    
    response = await react_handler(
        query,
        llm=kitchen.manager.get_dependency(DependencyType.LLM),
        system_prompt=kitchen.manager.get_dependency(DependencyType.SYSTEM_PROMPT)
    )
    
    assert response.token_counts is not None
    assert response.token_counts.llm_prompt_tokens > 0
    assert response.token_counts.llm_completion_tokens > 0
    assert response.token_counts.total_llm_tokens > 0

@pytest.mark.asyncio
async def test_tool_parsing():
    """Test tool call parsing"""
    text = """Thought: I should calculate this
Action: calculator
Input: 2 + 2"""
    
    from app.handlers.react import parse_tool_call
    result = parse_tool_call(text)
    
    assert result is not None
    assert result["tool"] == "calculator"
    assert result["input"] == "2 + 2" 