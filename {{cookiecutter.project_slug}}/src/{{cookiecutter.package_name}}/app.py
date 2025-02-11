from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.client import WhiskClient
import asyncio
from pathlib import Path
import click
from .config import Settings
from .utils.logging import logger
from .handlers import VectorStoreManager

# Initialize KitchenAI App
kitchen = KitchenAIApp(namespace="llama-index")

# Initialize handlers based on what's available
try:
    from .handlers import QueryHandler
    QueryHandler(kitchen)
except ImportError:
    pass

try:
    from .handlers import StorageHandler
    StorageHandler(kitchen)
except ImportError:
    pass

try:
    from .handlers import EmbedHandler
    EmbedHandler(kitchen)
except ImportError:
    pass

def create_client(config_path: Path = None) -> WhiskClient:
    """Create a WhiskClient instance with the configured kitchen."""
    settings = Settings.from_config(config_path)
    return WhiskClient(
        nats_url=settings.nats_url,
        client_id=settings.client_id,
        user=settings.nats_user,
        password=settings.nats_password,
        kitchen=kitchen,
    )

async def start_client(client: WhiskClient):
    """Start the WhiskClient."""
    await client.run()

@click.command()
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Path to config file')
def main(config: Path = None):
    """Run the application."""
    try:
        client = create_client(config)
        asyncio.run(start_client(client))
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Error running client: {e}")
        raise

if __name__ == "__main__":
    main() 