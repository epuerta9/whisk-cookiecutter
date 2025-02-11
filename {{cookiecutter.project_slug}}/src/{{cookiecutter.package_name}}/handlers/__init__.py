"""Whisk handlers initialization."""
from .vector_store import VectorStoreManager

# Build __all__ list dynamically based on what's available
__all__ = ['VectorStoreManager']

try:
    from .query import QueryHandler
    __all__.append('QueryHandler')
except ImportError:
    pass

try:
    from .storage import StorageHandler
    __all__.append('StorageHandler')
except ImportError:
    pass

try:
    from .embed import EmbedHandler
    __all__.append('EmbedHandler')
except ImportError:
    pass 