from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.layout import Layout
from typing import List, Dict

class KivaTUI:
    def __init__(self):
        self.console = Console()
        self.live = Live(console=self.console, refresh_per_second=10, screen=False, transient=True)
        self.status = "Initializing..."
        self.realtime_text = ""
        self.current_response = None  # None when not generating, str when generating
        self.is_streaming_response = False # Tracks if the assistant has started streaming its response
        self.title = "[bold green]Kiva Voice Assistant[/bold green]"
    
    def start(self):
        self.live.start()
        self.update_panel()
        
    def stop(self):
        if self.current_response is not None:
            self.finalize_assistant_message()
        self.live.stop()

    def set_status(self, status: str):
        self.status = status
        self.update_panel()

    def set_realtime_text(self, text: str):
        self.realtime_text = text
        self.update_panel()

    def add_user_message(self, text: str):
        # Print directly to the console (above the Live display)
        self.live.console.print(
            Panel(Text(text, style="cyan"), title="[bold cyan]You[/bold cyan]", border_style="cyan")
        )
        self.realtime_text = "" # Clear realtime text
        self.update_panel()

    def add_assistant_message(self, text: str):
        # Start tracking a new response
        self.current_response = text
        self.is_streaming_response = False # Reset streaming flag for a new response
        self.update_panel()

    def add_system_message(self, text: str):
        # Print system messages directly to log
        self.live.console.print(Text(f"[System] {text}", style="dim white"))

    def update_last_assistant_message(self, text_chunk: str):
        """Appends text to the current assistant message for streaming."""
        if self.current_response is not None:
            self.current_response += text_chunk
            if text_chunk: # Only set to True if a non-empty chunk is received
                self.is_streaming_response = True
            self.update_panel()
        else:
            self.add_assistant_message(text_chunk)

    def finalize_assistant_message(self):
        """Commits the current streaming response to the permanent log."""
        if self.current_response:
            self.live.console.print(
                Panel(Markdown(self.current_response), title="[bold magenta]Kiva[/bold magenta]", border_style="magenta")
            )
            self.current_response = None
            self.is_streaming_response = False # Reset after finalizing
            self.update_panel()

    def update_panel(self):
        items = []

        # 1. If we are generating a response, show it LIVE
        if self.current_response is not None:
            content = self.current_response if self.current_response else "..."
            
            # Change title based on streaming status
            if self.is_streaming_response:
                assistant_title = "[bold magenta]Kiva[/bold magenta]"
            else:
                assistant_title = "[bold magenta]Kiva (Thinking...)[/bold magenta]"

            items.append(
                Panel(
                    Markdown(content), 
                    title=assistant_title, 
                    border_style="magenta"
                )
            )

        # 2. Status Panel (Always visible at bottom)
        status_text = Text("Status: ", style="bold white")
        
        status_style = "green"
        if "Listening" in self.status:
            status_style = "green"
        elif "Speaking" in self.status:
            status_style = "blue"
        elif "Processing" in self.status:
            status_style = "yellow"
        
        status_text.append(self.status, style=status_style)
        
        # Add realtime transcription if available
        if self.realtime_text:
            status_text.append("\nListening: ", style="dim white")
            status_text.append(self.realtime_text, style="yellow")

        items.append(
            Panel(status_text, border_style="green", style="bold white", title=self.title)
        )
        
        # Update Live display with the Group
        self.live.update(Group(*items))

    def print_system(self, text: str):
        self.add_system_message(text) 
