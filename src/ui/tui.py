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
        self.live = Live(console=self.console, refresh_per_second=10, screen=True)
        # Store messages as dicts: {'role': 'user'|'assistant'|'system', 'content': 'text'}
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

    def add_system_message(self, text: str):
        # Don't add system messages to history for display
        pass

    def update_last_assistant_message(self, text_chunk: str):
        """Appends text to the last assistant message for streaming."""
        if self.history and self.history[-1]['role'] == 'assistant':
            self.history[-1]['content'] += text_chunk
            self.update_panel()
        else:
            # Fallback if no message exists yet
            self.add_assistant_message(text_chunk)

    def update_panel(self):
        # Create Layout
        layout = Layout()
        layout.split_column(
            Layout(name="body", ratio=1),
            Layout(name="status", size=3)
        )

        items = []
        
        # Filter out system messages for display
        display_history = [msg for msg in self.history if msg['role'] != 'system']
        
        # Show History (Last 12 visible messages)
        display_messages = display_history[-12:] 
        
        for msg in display_messages:
            if msg['role'] == 'user':
                items.append(Text("You: ", style="bold cyan"))
                items.append(Text(msg['content'], style="cyan"))
                items.append(Text("\n"))
            elif msg['role'] == 'assistant':
                items.append(Text("Kiva: ", style="bold magenta"))
                # Use Markdown for assistant messages
                # We need to ensure it doesn't break if empty
                content = msg['content'] if msg['content'] else "..."
                items.append(Markdown(content, style="magenta"))
                items.append(Text("\n"))
        
        # Show Real-time Text (User is typing/speaking)
        if self.realtime_text:
            items.append(Text("You: ", style="bold yellow"))
            items.append(Text(self.realtime_text, style="yellow"))
            items.append(Text("\n"))
        
        # Update Body Layout
        layout["body"].update(
            Panel(
                Group(*items), 
                title=self.title, 
                border_style="green",
                padding=(1, 2)
            )
        )
        
        # Update Status Layout
        status_text = Text("Status: ", style="bold white")
        
        status_style = "green"
        if "Listening" in self.status:
            status_style = "green"
        elif "Speaking" in self.status:
            status_style = "blue"
        elif "Processing" in self.status:
            status_style = "yellow"
        
        status_text.append(self.status, style=status_style)
        
        layout["status"].update(
            Panel(status_text, border_style="green", style="bold white")
        )
        
        self.live.update(layout)

    def print_system(self, text: str):
        """Print a system message above the live display."""
        # In live mode, printing directly might break layout, better to add as system message
        # or log to file. For now, just add to history or ignore if purely debug.
        pass 
