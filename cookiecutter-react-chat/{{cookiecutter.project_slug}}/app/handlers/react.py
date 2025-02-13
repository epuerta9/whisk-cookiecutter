from whisk.kitchenai_sdk.schema import (
    WhiskQuerySchema,
    WhiskQueryBaseResponseSchema,
    TokenCountSchema
)
from llama_index.core.callbacks import TokenCountingHandler
from typing import List, Dict, Any, Optional
import tiktoken
import json
import re

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

class Tool:
    def __init__(self, name: str, description: str, func: callable):
        self.name = name
        self.description = description
        self.func = func

    async def __call__(self, **kwargs) -> str:
        return await self.func(**kwargs)

# Example tool implementations
async def search(query: str) -> str:
    return f"Search results for: {query}"

async def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Error evaluating expression"

async def weather(location: str) -> str:
    return f"Weather for {location}: Sunny, 72°F"

# Available tools
TOOLS = {
    "search": Tool(
        "search",
        "Search for information on the internet",
        search
    ),
    "calculator": Tool(
        "calculator",
        "Evaluate mathematical expressions",
        calculator
    ),
    "weather": Tool(
        "weather",
        "Get weather information for a location",
        weather
    )
}

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse tool calls from text using regex"""
    pattern = r"Action: (\w+)\nInput: (.+)"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        return {
            "tool": match.group(1),
            "input": match.group(2).strip()
        }
    return None

async def react_handler(data: WhiskQuerySchema, llm=None, system_prompt=None) -> WhiskQueryBaseResponseSchema:
    """ReAct chat handler for tool-augmented responses.
    
    Args:
        data (WhiskQuerySchema): Query request with fields:
            - query (str): The user's message
            - label (str): Handler label (e.g. "react")
            - metadata (dict, optional): Additional context
            - stream (bool, optional): Enable streaming response
            - stream_id (str, optional): ID for streaming session
            - messages (list, optional): Chat history
        llm: Language model for generating responses
        system_prompt (str, optional): System prompt describing available tools
        
    Returns:
        WhiskQueryBaseResponseSchema: Response containing:
            - input (str): Original message
            - output (str): Generated response
            - metadata (dict): Response metadata including tool usage
            - token_counts (TokenCountSchema): Token usage stats
            - messages (list): Updated chat history
    """
    try:
        # Prepare chat history
        messages = data.messages or []
        
        # Add system prompt with tool descriptions
        if not messages:
            tool_descriptions = "\n".join([
                f"- {name}: {tool.description}"
                for name, tool in TOOLS.items()
            ])
            system_message = (
                f"{system_prompt}\n\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                "To use a tool, respond with:\n"
                "Thought: what you're thinking\n"
                "Action: tool_name\n"
                "Input: tool input\n\n"
                "After using a tool, I'll show you the result and you can continue thinking."
            )
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": data.query})
        
        # Track tool usage
        tool_usage = []
        max_steps = 5
        
        # ReAct loop
        for _ in range(max_steps):
            # Get next action from LLM
            response = await llm.acomplete(
                messages=messages,
                callbacks=[token_counter]
            )
            
            # Parse tool call
            tool_call = parse_tool_call(response.response)
            
            if tool_call and tool_call["tool"] in TOOLS:
                # Execute tool
                tool = TOOLS[tool_call["tool"]]
                result = await tool(**{"query": tool_call["input"]})
                
                # Track usage
                tool_usage.append({
                    "tool": tool_call["tool"],
                    "input": tool_call["input"],
                    "output": result
                })
                
                # Add to conversation
                messages.append({"role": "assistant", "content": response.response})
                messages.append({"role": "system", "content": f"Tool result: {result}"})
            else:
                # Final response
                messages.append({"role": "assistant", "content": response.response})
                break
        
        # Get token counts
        token_counts = TokenCountSchema(
            llm_prompt_tokens=token_counter.prompt_llm_token_count,
            llm_completion_tokens=token_counter.completion_llm_token_count,
            total_llm_tokens=token_counter.total_llm_token_count
        )
        token_counter.reset_counts()
        
        # Prepare metadata
        metadata = {
            "token_counts": token_counts.dict(),
            "tool_usage": tool_usage
        }
        if data.metadata:
            metadata.update(data.metadata)
            
        return WhiskQueryBaseResponseSchema(
            input=data.query,
            output=response.response,
            metadata=metadata,
            token_counts=token_counts,
            messages=messages
        )
            
    except Exception as e:
        return WhiskQueryBaseResponseSchema(
            input=data.query,
            output=f"Error: {str(e)}",
            metadata=data.metadata,
            token_counts=TokenCountSchema(),
            messages=data.messages
        ) 