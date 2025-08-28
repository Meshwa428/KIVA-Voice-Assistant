# backend/mcp_servers/vscode_mcp.py

from mcp.server.fastmcp import FastMCP
import subprocess
import os

mcp = FastMCP("VSCode")

@mcp.tool()
def open_project(path: str) -> str:
    """
    Opens a project or folder in VS Code.
    For security, this is restricted to a predefined list of allowed project directories.
    """
    # For security, we should have a whitelist of allowed project directories.
    # For now, let's assume a base directory for projects.
    ALLOWED_PROJECTS_DIR = os.path.abspath("projects")
    if not os.path.exists(ALLOWED_PROJECTS_DIR):
        os.makedirs(ALLOWED_PROJECTS_DIR)

    project_path = os.path.abspath(os.path.join(ALLOWED_PROJECTS_DIR, path))

    if os.path.commonpath([project_path, ALLOWED_PROJECTS_DIR]) != ALLOWED_PROJECTS_DIR:
        return "Error: Access denied. Project is outside the allowed directory."

    if not os.path.isdir(project_path):
        return f"Error: Project directory not found at {path}"

    try:
        subprocess.run(["code", project_path], check=True)
        return f"Opened project: {path}"
    except FileNotFoundError:
        return "Error: 'code' command not found. Is VS Code installed and in your PATH?"
    except subprocess.CalledProcessError as e:
        return f"Error opening project: {e}"

@mcp.tool()
def open_file(path: str) -> str:
    """
    Opens a file in VS Code.
    The path is relative to a safe base directory.
    """
    # This can be dangerous. Let's restrict it to the same projects directory for now.
    ALLOWED_FILES_DIR = os.path.abspath("projects")

    file_path = os.path.abspath(os.path.join(ALLOWED_FILES_DIR, path))

    if os.path.commonpath([file_path, ALLOWED_FILES_DIR]) != ALLOWED_FILES_DIR:
        return "Error: Access denied. File is outside the allowed directory."

    if not os.path.isfile(file_path):
        return f"Error: File not found at {path}"

    try:
        subprocess.run(["code", file_path], check=True)
        return f"Opened file: {path}"
    except FileNotFoundError:
        return "Error: 'code' command not found. Is VS Code installed and in your PATH?"
    except subprocess.CalledProcessError as e:
        return f"Error opening file: {e}"

# To run this server:
# uvicorn backend.mcp_servers.vscode_mcp:mcp.app --port 8004
