# backend/mcp_servers/github_mcp.py

from mcp.server.fastmcp import FastMCP
from github import Github
import os

mcp = FastMCP("GitHub")

# --- Authentication ---
# IMPORTANT: You need to create a Personal Access Token (PAT) on GitHub
# with the necessary permissions (e.g., repo) and set it as an environment variable.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

g = None

@mcp.app.on_event("startup")
async def startup_event():
    global g
    if GITHUB_TOKEN:
        g = Github(GITHUB_TOKEN)
        print("GitHub MCP server started.")
    else:
        print("Warning: GITHUB_TOKEN environment variable not set. GitHub tools will not work.")

@mcp.tool()
def list_prs(repo_full_name: str) -> list[str]:
    """
    Lists pull requests for a given repository.
    repo_full_name should be in the format 'owner/repo'.
    """
    if not g:
        return ["Error: GitHub not authenticated. GITHUB_TOKEN not set."]
    try:
        repo = g.get_repo(repo_full_name)
        prs = repo.get_pulls(state='open', sort='created', direction='desc')
        return [f"#{pr.number}: {pr.title}" for pr in prs]
    except Exception as e:
        return [f"Error listing PRs: {e}"]

@mcp.tool()
def create_issue(repo_full_name: str, title: str, body: str) -> str:
    """
    Creates an issue in a given repository.
    repo_full_name should be in the format 'owner/repo'.
    """
    if not g:
        return "Error: GitHub not authenticated. GITHUB_TOKEN not set."
    try:
        repo = g.get_repo(repo_full_name)
        issue = repo.create_issue(title=title, body=body)
        return f"Successfully created issue #{issue.number}: {issue.html_url}"
    except Exception as e:
        return f"Error creating issue: {e}"

@mcp.tool()
def get_pr_details(repo_full_name: str, pr_number: int) -> str:
    """
    Gets the details of a specific pull request.
    repo_full_name should be in the format 'owner/repo'.
    """
    if not g:
        return "Error: GitHub not authenticated. GITHUB_TOKEN not set."
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        return f"PR #{pr.number}: {pr.title}\nURL: {pr.html_url}\nState: {pr.state}\nBody:\n{pr.body}"
    except Exception as e:
        return f"Error getting PR details: {e}"

# To run this server:
# uvicorn backend.mcp_servers.github_mcp:mcp.app --port 8005
