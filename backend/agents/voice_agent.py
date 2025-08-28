# backend/agents/voice_agent.py

import requests
from langchain.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Define the tools for the file system MCP server
@tool
def read_file(path: str) -> str:
    """
    Reads the content of a file.
    The path is relative to the safe base directory of the file server.
    """
    response = requests.post("http://localhost:8001/tools/read_file/run", json={"path": path})
    return response.json()["output"]

@tool
def write_file(path: str, content: str) -> str:
    """
    Writes content to a file.
    The path is relative to the safe base directory of the file server.
    """
    response = requests.post("http://localhost:8001/tools/write_file/run", json={"path": path, "content": content})
    return response.json()["output"]

@tool
def list_files(path: str = ".") -> list[str]:
    """
    Lists files and directories in a given path.
    The path is relative to the safe base directory of the file server.
    """
    response = requests.post("http://localhost:8001/tools/list_files/run", json={"path": path})
    return response.json()["output"]


# Define the tools for the browser MCP server
@tool
def open_url(url: str) -> str:
    """
    Opens a URL in the browser.
    """
    response = requests.post("http://localhost:8002/tools/open_url/run", json={"url": url})
    return response.json()["output"]

@tool
def search(query: str) -> str:
    """
    Performs a web search.
    """
    response = requests.post("http://localhost:8002/tools/search/run", json={"query": query})
    return response.json()["output"]

@tool
def click(selector: str) -> str:
    """
    Clicks on an element in the browser page using a CSS selector.
    """
    response = requests.post("http://localhost:8002/tools/click/run", json={"selector": selector})
    return response.json()["output"]


# Define the tools for the Spotify MCP server
@tool
def play_track(track: str = None) -> str:
    """
    Plays a track on Spotify. If no track is provided, resumes playback.
    """
    response = requests.post("http://localhost:8003/tools/play/run", json={"track": track})
    return response.json()["output"]

@tool
def pause_playback() -> str:
    """
    Pauses the current playback on Spotify.
    """
    response = requests.post("http://localhost:8003/tools/pause/run")
    return response.json()["output"]

@tool
def next_track() -> str:
    """
    Skips to the next track on Spotify.
    """
    response = requests.post("http://localhost:8003/tools/next_track/run")
    return response.json()["output"]

@tool
def previous_track() -> str:
    """
    Goes to the previous track on Spotify.
    """
    response = requests.post("http://localhost:8003/tools/previous_track/run")
    return response.json()["output"]

@tool
def get_current_track() -> str:
    """
    Gets the currently playing track on Spotify.
    """
    response = requests.post("http://localhost:8003/tools/get_current_track/run")
    return response.json()["output"]


# Define the tools for the VS Code MCP server
@tool
def open_project(path: str) -> str:
    """
    Opens a project or folder in VS Code.
    """
    response = requests.post("http://localhost:8004/tools/open_project/run", json={"path": path})
    return response.json()["output"]

@tool
def open_file(path: str) -> str:
    """
    Opens a file in VS Code.
    """
    response = requests.post("http://localhost:8004/tools/open_file/run", json={"path": path})
    return response.json()["output"]


# Define the tools for the GitHub MCP server
@tool
def list_prs(repo_full_name: str) -> list[str]:
    """
    Lists pull requests for a given repository.
    repo_full_name should be in the format 'owner/repo'.
    """
    response = requests.post("http://localhost:8005/tools/list_prs/run", json={"repo_full_name": repo_full_name})
    return response.json()["output"]

@tool
def create_issue(repo_full_name: str, title: str, body: str) -> str:
    """
    Creates an issue in a given repository.
    repo_full_name should be in the format 'owner/repo'.
    """
    response = requests.post("http://localhost:8005/tools/create_issue/run", json={"repo_full_name": repo_full_name, "title": title, "body": body})
    return response.json()["output"]

@tool
def get_pr_details(repo_full_name: str, pr_number: int) -> str:
    """
    Gets the details of a specific pull request.
    repo_full_name should be in the format 'owner/repo'.
    """
    response = requests.post("http://localhost:8005/tools/get_pr_details/run", json={"repo_full_name": repo_full_name, "pr_number": pr_number})
    return response.json()["output"]


def create_voice_agent():
    """
    Creates and returns a LangChain agent with the file system and browser tools.
    """
    tools = [
        read_file, write_file, list_files,
        open_url, search, click,
        play_track, pause_playback, next_track, previous_track, get_current_track,
        open_project, open_file,
        list_prs, create_issue, get_pr_details
    ]

    # Use a pre-built ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    llm = ChatOllama(model="llama2")

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

if __name__ == '__main__':
    # This is for standalone testing of the agent.
    # It requires the MCP servers to be running.
    # uvicorn backend.mcp_servers.files_mcp:mcp.app --port 8001
    # uvicorn backend.mcp_servers.browser_mcp:mcp.app --port 8002

    # agent_executor = create_voice_agent()

    # Example usage:
    # result = agent_executor.invoke({"input": "list the files in the current directory"})
    # print(result)

    # result = agent_executor.invoke({"input": "open the url https://www.google.com"})
    # print(result)

    print("Voice agent created. This will be used by the main voice assistant.")
