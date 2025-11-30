from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from typing import List, Dict

class KivaTUI:
    def __init__(self):
        self.console = Console()
        self.live = Live(console=self.console, refresh_per_second=10, screen=False)
        # Store messages as dicts: {'role': 'user'|'assistant', 'content': 'text'}
        self.history: List[Dict[str, str]] = []
        self.status = "Initializing..."
        self.realtime_text = ""
        self.title = "[bold green]Kiva Voice Assistant[/bold green]"
    
    def start(self):
        self.live.start()
        self.update_panel()
        
    def stop(self):
        self.live.stop()

    def set_status(self, status: str):
        self.status = status
        self.update_panel()

    def set_realtime_text(self, text: str):
        self.realtime_text = text
        self.update_panel()

    def add_user_message(self, text: str):
        self.history.append({'role': 'user', 'content': text})
        self.realtime_text = "" # Clear realtime text
        self.update_panel()

    def add_assistant_message(self, text: str):
        self.history.append({'role': 'assistant', 'content': text})
        self.update_panel()

    def update_last_assistant_message(self, text_chunk: str):
        """Appends text to the last assistant message for streaming."""
        if self.history and self.history[-1]['role'] == 'assistant':
            self.history[-1]['content'] += text_chunk
            self.update_panel()
        else:
            # Fallback if no message exists yet
            self.add_assistant_message(text_chunk)

    def update_panel(self):
        content = Text()
        
        # Show History (Last 6 lines)
        display_messages = self.history[-6:] 
        
        for msg in display_messages:
            if msg['role'] == 'user':
                content.append("You: ", style="bold cyan")
                content.append(msg['content'] + "\n\n", style="cyan")
            else:
                content.append("Kiva: ", style="bold magenta")
                content.append(msg['content'] + "\n\n", style="magenta")
        
        # Show Real-time Text (User is typing/speaking)
        if self.realtime_text:
            content.append("You: ", style="bold yellow")
            content.append(self.realtime_text, style="yellow")
        
        # Footer / Status Line
        content.append("\n\n")
        content.append("Status: ", style="bold white")
        
        status_style = "green"
        if "Listening" in self.status:
            status_style = "green"
        elif "Speaking" in self.status:
            status_style = "blue"
        elif "Processing" in self.status:
            status_style = "yellow"
        
        content.append(self.status, style=status_style)
        
        panel = Panel(
            content, 
            title=self.title, 
            border_style="green",
            padding=(1, 2)
        )
        self.live.update(panel)

    def print_system(self, text: str):
        """Print a system message above the live display."""
        self.console.print(text)
