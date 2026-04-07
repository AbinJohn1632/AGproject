import os
import sys

def check_virtualenv() -> bool:
    """Check if the python instance is running in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def setup_directories(base_path: str):
    """Ensure essential directories exist."""
    dirs = ['vectordb']
    for d in dirs:
        full_path = os.path.join(base_path, d)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

def estimate_tokens(text: str) -> int:
    """A simplistic token estimator (roughly 4 characters per token for English)."""
    return len(text) // 4
