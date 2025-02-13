import tiktoken
from llama_index.core.callbacks import TokenCountingHandler

def create_token_counter():
    """Create token counter for tracking usage"""
    return TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    ) 