from whisk.kitchenai_sdk.schema import (
    WhiskQuerySchema,
    WhiskQueryBaseResponseSchema,
    TokenCountSchema
)
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from llama_index.core.callbacks import TokenCountingHandler
import tiktoken
from typing import List, Dict, Any, Optional

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

class MemoryManager:
    def __init__(self, memory_type: str = "{{ cookiecutter.memory_type }}", k: int = {{ cookiecutter.memory_k }}):
        self.memory_type = memory_type
        self.k = k
        self.memory = self._create_memory()
        
    def _create_memory(self):
        if self.memory_type == "buffer":
            return ConversationBufferMemory()
        elif self.memory_type == "window":
            return ConversationBufferWindowMemory(k=self.k)
        elif self.memory_type == "summary":
            return ConversationSummaryMemory()
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")
            
    def add_message(self, message: str, is_human: bool = True):
        if is_human:
            self.memory.chat_memory.add_message(HumanMessage(content=message))
        else:
            self.memory.chat_memory.add_message(AIMessage(content=message))
            
    def get_history(self) -> List[Dict[str, str]]:
        return [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant",
             "content": msg.content}
            for msg in self.memory.chat_memory.messages
        ]
        
    def clear(self):
        self.memory.clear()

# Initialize memory manager
memory_manager = MemoryManager()

async def memory_handler(data: WhiskQuerySchema, llm=None, system_prompt=None) -> WhiskQueryBaseResponseSchema:
    """Memory-based chat handler using Langchain memory types.
    
    Args:
        data (WhiskQuerySchema): Query request with fields:
            - query (str): The user's message
            - label (str): Handler label (e.g. "memory")
            - metadata (dict, optional): Additional context
            - stream (bool, optional): Enable streaming response
            - stream_id (str, optional): ID for streaming session
            - messages (list, optional): Chat history
        llm: Language model for generating responses
        system_prompt (str, optional): System prompt for the conversation
        
    Returns:
        WhiskQueryBaseResponseSchema: Response containing:
            - input (str): Original message
            - output (str): Generated response
            - metadata (dict): Response metadata including memory info
            - token_counts (TokenCountSchema): Token usage stats
            - messages (list): Updated chat history
    """
    try:
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add chat history from memory
        messages.extend(memory_manager.get_history())
        
        # Add current message
        messages.append({"role": "user", "content": data.query})
        memory_manager.add_message(data.query, is_human=True)
        
        # Get response from LLM
        response = await llm.acomplete(
            messages=messages,
            callbacks=[token_counter]
        )
        
        # Add response to memory
        memory_manager.add_message(response.response, is_human=False)
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
            "memory_type": memory_manager.memory_type,
            "memory_size": len(memory_manager.get_history())
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

async def clear_memory_handler(data: WhiskQuerySchema) -> WhiskQueryBaseResponseSchema:
    """Handler to clear conversation memory."""
    try:
        memory_manager.clear()
        return WhiskQueryBaseResponseSchema(
            input=data.query,
            output="Memory cleared successfully",
            metadata={"memory_type": memory_manager.memory_type, "memory_size": 0}
        )
    except Exception as e:
        return WhiskQueryBaseResponseSchema(
            input=data.query,
            output=f"Error clearing memory: {str(e)}"
        ) 