import os
from dotenv import load_dotenv
from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import DependencyType

from .dependencies.llm import setup_llm
from .utils.token_counter import create_token_counter
from .handlers import chat

# Load environment variables
load_dotenv()

# Initialize token counter
token_counter = create_token_counter()

# Initialize dependencies
llm = setup_llm(token_counter)

# Initialize KitchenAI App
kitchen = KitchenAIApp(namespace="{{ cookiecutter.project_slug }}")

# Register dependencies
kitchen.register_dependency(DependencyType.LLM, llm)
kitchen.register_dependency(DependencyType.SYSTEM_PROMPT, "{{ cookiecutter.system_prompt }}")

# Register handlers
kitchen.query.handler("chat", DependencyType.LLM, DependencyType.SYSTEM_PROMPT)(
    chat.chat_handler
)

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