# backend/mcp_servers/spotify_mcp.py

from mcp.server.fastmcp import FastMCP
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

mcp = FastMCP("Spotify")

# --- Authentication ---
# IMPORTANT: You need to register an application on the Spotify Developer Dashboard
# to get a client ID and client secret.
# You also need to set a redirect URI in your application settings on the dashboard.
# For a local server, http://localhost:8003/callback is a good choice.

CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID", "YOUR_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIPY_REDIRECT_URI", "http://localhost:8003/callback")

# Scope for controlling playback
SCOPE = "user-modify-playback-state user-read-playback-state"

sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope=SCOPE)

sp = None # Will be initialized after authentication

@mcp.app.on_event("startup")
async def startup_event():
    # In a real application, you would need a proper authentication flow here.
    # For example, you could print an auth URL for the user to visit,
    # and then handle the callback to get the token.
    # For this implementation, we will assume the token is available
    # (e.g., from a previous run or a cached token).

    global sp
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        # This would be the place to initiate the auth flow
        print("Spotify authentication required. Please run the auth flow.")
        # For now, we can't proceed without a token.
        return

    sp = spotipy.Spotify(auth=token_info['access_token'])
    print("Spotify MCP server started.")


@mcp.tool()
def play(track: str = None):
    """
    Plays a track on Spotify. If no track is provided, resumes playback.
    """
    if not sp: return "Error: Spotify not authenticated."
    try:
        if track:
            results = sp.search(q=track, type='track', limit=1)
            if results['tracks']['items']:
                track_uri = results['tracks']['items'][0]['uri']
                sp.start_playback(uris=[track_uri])
                return f"Playing {track}"
            else:
                return f"Track '{track}' not found."
        else:
            sp.start_playback()
            return "Playback resumed."
    except Exception as e:
        return f"Error playing: {e}"

@mcp.tool()
def pause():
    """
    Pauses the current playback.
    """
    if not sp: return "Error: Spotify not authenticated."
    try:
        sp.pause_playback()
        return "Playback paused."
    except Exception as e:
        return f"Error pausing: {e}"

@mcp.tool()
def next_track():
    """
    Skips to the next track.
    """
    if not sp: return "Error: Spotify not authenticated."
    try:
        sp.next_track()
        return "Skipped to next track."
    except Exception as e:
        return f"Error skipping track: {e}"

@mcp.tool()
def previous_track():
    """
    Goes to the previous track.
    """
    if not sp: return "Error: Spotify not authenticated."
    try:
        sp.previous_track()
        return "Went to previous track."
    except Exception as e:
        return f"Error going to previous track: {e}"

@mcp.tool()
def get_current_track() -> str:
    """
    Gets the currently playing track.
    """
    if not sp: return "Error: Spotify not authenticated."
    try:
        track_info = sp.current_playback()
        if track_info and track_info['is_playing']:
            item = track_info['item']
            return f"Currently playing: {item['name']} by {', '.join(artist['name'] for artist in item['artists'])}"
        else:
            return "Nothing is currently playing."
    except Exception as e:
        return f"Error getting current track: {e}"

# To run this server:
# uvicorn backend.mcp_servers.spotify_mcp:mcp.app --port 8003
