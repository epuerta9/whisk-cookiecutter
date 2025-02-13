from whisk.kitchenai_sdk.schema import (
    WhiskQuerySchema,
    WhiskQueryBaseResponseSchema,
    TokenCountSchema
)
from llama_index.core.callbacks import TokenCountingHandler
import tiktoken

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

async def chat_handler(data: WhiskQuerySchema, llm=None, system_prompt=None) -> WhiskQueryBaseResponseSchema:
    """Chat handler for personality-based responses.
    
    Args:
        data (WhiskQuerySchema): Query request with fields:
            - query (str): The user's message
            - label (str): Handler label (e.g. "chat")
            - metadata (dict, optional): Additional context
            - stream (bool, optional): Enable streaming response
            - stream_id (str, optional): ID for streaming session
            - messages (list, optional): Chat history
        llm: Language model for generating responses
        system_prompt (str, optional): Personality system prompt
        
    Returns:
        WhiskQueryBaseResponseSchema: Response containing:
            - input (str): Original message
            - output (str): Generated response
            - metadata (dict): Response metadata
            - token_counts (TokenCountSchema): Token usage stats
            - messages (list): Updated chat history
    """
    try:
        # Prepare chat history
        messages = data.messages or []
        
        # Add system prompt if provided
        if system_prompt and not messages:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add user message
        messages.append({"role": "user", "content": data.query})
        
        # Get response from LLM
        response = await llm.acomplete(
            messages=messages,
            callbacks=[token_counter]
        )
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.response})
        
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
            "personality": "{{ cookiecutter.personality }}"
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