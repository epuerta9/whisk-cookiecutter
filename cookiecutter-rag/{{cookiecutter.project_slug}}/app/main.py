import os
from pathlib import Path
from dotenv import load_dotenv
from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import DependencyType
from llama_index.core.prompts.system import SHAKESPEARE_WRITING_ASSISTANT

from .dependencies.llm import setup_llm
from .dependencies.vector_store import setup_vector_store
from .utils.token_counter import create_token_counter
from .handlers import query, storage

# Load environment variables
load_dotenv()

# Initialize token counter
token_counter = create_token_counter()

# Initialize dependencies
llm = setup_llm(token_counter)

# Setup vector store with string path
chroma_path = os.path.join(os.getcwd(), "chroma_db")
vector_store = setup_vector_store(chroma_path)

# Initialize KitchenAI App
kitchen = KitchenAIApp(namespace="{{ cookiecutter.project_slug }}")

# Register dependencies
kitchen.register_dependency(DependencyType.LLM, llm)
kitchen.register_dependency(DependencyType.VECTOR_STORE, vector_store)
kitchen.register_dependency(DependencyType.SYSTEM_PROMPT, SHAKESPEARE_WRITING_ASSISTANT)

# Register handlers
kitchen.query.handler("query", DependencyType.LLM, DependencyType.VECTOR_STORE, DependencyType.SYSTEM_PROMPT)(
    query.query_handler
)
kitchen.storage.handler("storage")(storage.storage_handler)
kitchen.storage.on_delete("storage")(storage.storage_delete_handler)

if __name__ == "__main__":
    from whisk.client import WhiskClient
    import asyncio
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    client = WhiskClient(
        nats_url=os.getenv("WHISK_NATS_URL", "nats://nats.playground.kitchenai.dev"),
        client_id=os.getenv("WHISK_CLIENT_ID", "whisk_client"),
        user=os.getenv("WHISK_NATS_USER", "playground"),
        password=os.getenv("WHISK_NATS_PASSWORD", "kitchenai_playground"),
        kitchen=kitchen,
    )
    
    async def start():
        await client.run()

    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...") 