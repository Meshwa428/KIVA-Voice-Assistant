# Textual UI

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.containers import Container
from backend.app import VoiceAssistant
import threading

class VoiceAssistantTUI(App):
    """
    A Textual TUI for the Voice Assistant.
    """
    CSS_PATH = "tui.css"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.assistant = VoiceAssistant(
            on_transcript=self.handle_transcript,
            on_llm_token=self.handle_llm_token
        )
        self.llm_full_response = ""

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="main_container"):
            with Container(id="left_pane"):
                yield Static("Live Transcript", id="transcript_title")
                yield Static("", id="transcript_display")
            with Container(id="right_pane"):
                yield Static("LLM Output", id="llm_title")
                yield Static("", id="llm_display")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.assistant_thread = threading.Thread(target=self.assistant.run)
        self.assistant_thread.daemon = True
        self.assistant_thread.start()
        self.query_one("#transcript_display").update("Voice assistant started. Speak now.")

    def handle_transcript(self, text: str):
        """
        Callback to handle the transcript from the assistant.
        This is called from a different thread.
        """
        # Clear the LLM display for the new response
        self.llm_full_response = ""
        self.call_from_thread(self.query_one("#llm_display").update, "")

        self.call_from_thread(self.query_one("#transcript_display").update, text)

    def handle_llm_token(self, token: str):
        """
        Callback to handle LLM tokens from the assistant.
        This is called from a different thread.
        """
        self.llm_full_response += token
        self.call_from_thread(self.query_one("#llm_display").update, self.llm_full_response)


if __name__ == "__main__":
    app = VoiceAssistantTUI()
    app.run()
