from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager

def setup_llm(token_counter):
    """Initialize and configure LLM"""
    Settings.callback_manager = CallbackManager([token_counter])
    llm = OpenAI(model="gpt-3.5-turbo")
    Settings.llm = llm
    return llm 