# Textual UI

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Button
from textual.containers import Container
from textual.screen import ModalScreen
from backend.app import VoiceAssistant
import threading

class ConfirmationDialog(ModalScreen):
    """A modal dialog to ask for confirmation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self.message, id="dialog_message"),
            Container(
                Button("Approve", variant="primary", id="approve"),
                Button("Deny", variant="error", id="deny"),
                id="dialog_buttons",
            ),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self.dismiss(True)
        else:
            self.dismiss(False)

class VoiceAssistantTUI(App):
    """
    A Textual TUI for the Voice Assistant.
    """
    CSS_PATH = "tui.css"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.assistant = VoiceAssistant(
            on_transcript=self.handle_transcript,
            on_llm_token=self.handle_llm_token,
            request_approval=self.request_approval
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

    async def request_approval(self, message: str) -> bool:
        """
        Requests approval from the user via a modal dialog.
        """
        dialog = ConfirmationDialog(message)
        result = await self.push_screen_wait(dialog)
        return result

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
