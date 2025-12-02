import yaml
import os

def load_config(path: str = "config.yaml"):  # pyright: ignore[reportAny]
    if not os.path.exists(path):
        # Fallback to looking one directory up if running from src
        if os.path.exists(os.path.join("..", path)):
            path = os.path.join("..", path)
        else:
            raise FileNotFoundError(f"Config file not found: {path}")
            
    with open(path, "r") as f:
        return yaml.safe_load(f)  # pyright: ignore[reportAny]