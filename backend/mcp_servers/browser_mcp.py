# backend/mcp_servers/browser_mcp.py

from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
import asyncio

mcp = FastMCP("Browser")

browser = None
page = None

@mcp.app.on_event("startup")
async def startup_event():
    global browser, page
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch()
    page = await browser.new_page()
    print("Browser MCP server started, browser launched.")

@mcp.app.on_event("shutdown")
async def shutdown_event():
    if browser:
        await browser.close()
    print("Browser MCP server shut down, browser closed.")

@mcp.tool()
async def open_url(url: str) -> str:
    """
    Opens a URL in the browser.
    """
    if not page:
        return "Error: Browser not initialized."
    try:
        await page.goto(url)
        return f"Successfully opened URL: {url}"
    except Exception as e:
        return f"Error opening URL: {e}"

@mcp.tool()
async def search(query: str) -> str:
    """
    Performs a web search using Google.
    """
    if not page:
        return "Error: Browser not initialized."
    try:
        await page.goto(f"https://www.google.com/search?q={query}")
        return f"Search for '{query}' performed successfully."
    except Exception as e:
        return f"Error during search: {e}"

@mcp.tool()
async def click(selector: str) -> str:
    """
    Clicks on an element in the browser page using a CSS selector.
    """
    if not page:
        return "Error: Browser not initialized."
    try:
        await page.click(selector)
        return f"Clicked on element with selector: {selector}"
    except Exception as e:
        return f"Error clicking element: {e}"

# To run this server, you would typically use an ASGI server like uvicorn:
# uvicorn backend.mcp_servers.browser_mcp:mcp.app --port 8002
# This would be managed by the main application's startup process.
