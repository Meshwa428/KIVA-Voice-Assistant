# backend/mcp_servers/files_mcp.py

from mcp.server.fastmcp import FastMCP
import os

# Create an MCP server for file system operations
mcp = FastMCP("FileSystem")

# Define a safe base directory for file operations
# For now, this is just a subdirectory in the project.
# In a real application, this should be a more isolated location.
BASE_DIR = "mcp_filesystem"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

def is_safe_path(path):
    """
    Checks if the path is safe and within the BASE_DIR.
    Prevents directory traversal attacks.
    """
    abs_path = os.path.abspath(os.path.join(BASE_DIR, path))
    abs_base_dir = os.path.abspath(BASE_DIR)
    return os.path.commonpath([abs_path, abs_base_dir]) == abs_base_dir

@mcp.tool()
def read_file(path: str) -> str:
    """
    Reads the content of a file.
    The path is relative to the safe base directory.
    """
    if not is_safe_path(path):
        return "Error: Access denied. Path is outside the safe directory."

    try:
        full_path = os.path.join(BASE_DIR, path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {path}"
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Writes content to a file.
    The path is relative to the safe base directory.
    If the file exists, it will be overwritten.
    """
    if not is_safe_path(path):
        return "Error: Access denied. Path is outside the safe directory."

    try:
        full_path = os.path.join(BASE_DIR, path)
        # Ensure subdirectories exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"File written successfully to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

@mcp.tool()
def list_files(path: str = ".") -> list[str]:
    """
    Lists files and directories in a given path.
    The path is relative to the safe base directory.
    """
    if not is_safe_path(path):
        return ["Error: Access denied. Path is outside the safe directory."]

    try:
        full_path = os.path.join(BASE_DIR, path)
        return os.listdir(full_path)
    except FileNotFoundError:
        return [f"Error: Directory not found at {path}"]
    except Exception as e:
        return [f"Error listing files: {e}"]

# To run this server, you would typically use an ASGI server like uvicorn:
# uvicorn backend.mcp_servers.files_mcp:mcp.app --port 8001
# This would be managed by the main application's startup process.
