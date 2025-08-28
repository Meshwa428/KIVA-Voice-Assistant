# FastAPI gateway, MCP adapters

import asyncio
import ollama
import threading
from audio.stt_client import STTClient
from audio.tts_client import TTSClient

class VoiceAssistant:
    """
    The main voice assistant class that orchestrates STT, LLM, and TTS.
    """
    def __init__(self, model: str = "llama2", on_transcript=None, on_llm_token=None):
        """
        Initializes the VoiceAssistant.
        """
        self.stt_client = STTClient()
        self.tts_client = TTSClient(on_character=self.handle_llm_token)
        self.model = model
        self.loop = asyncio.get_event_loop()
        self.stt_thread = None
        self.on_transcript = on_transcript
        self.on_llm_token = on_llm_token
        print("Voice Assistant initialized.")

    def handle_llm_token(self, token: str):
        """
        Handles an LLM token from the TTS client.
        """
        if self.on_llm_token:
            self.on_llm_token(token)

    async def handle_transcript(self, text: str):
        """
        Handles the transcribed text from the STT client.
        """
        if self.on_transcript:
            self.on_transcript(text)

        # Simple command routing placeholder
        if text.lower().strip() == "hello":
            self.tts_client.feed("Hello to you too!")
            self.tts_client.play_async()
            return

        print("Sending to Ollama for chat...")

        try:
            async def response_generator():
                full_response = ""
                async for chunk in await ollama.AsyncClient().chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': text}],
                    stream=True
                ):
                    if content := chunk['message']['content']:
                        full_response += content
                        yield content
                print(f"Ollama full response: {full_response}")

            self.tts_client.feed(response_generator())
            self.tts_client.play_async()
        except Exception as e:
            print(f"Error during Ollama streaming or TTS playback: {e}")

    def _on_final_transcript(self, text: str):
        """
        Thread-safe callback for when a final transcript is received.
        """
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.handle_transcript(text), self.loop)

    def run(self):
        """
        Starts the voice assistant.
        """
        print("Starting Voice Assistant...")

        self.stt_thread = threading.Thread(target=self.stt_client.start, args=(self._on_final_transcript,))
        self.stt_thread.daemon = True
        self.stt_thread.start()

        print("Voice Assistant is running. Press Ctrl+C to exit.")
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            print("Shutting down Voice Assistant.")
        finally:
            self.loop.close()

if __name__ == '__main__':
    # This script is intended to be run as part of a larger application
    # with a running asyncio event loop (e.g., a Textual TUI or FastAPI server).

    print("Backend app structure created. This will be orchestrated by a TUI or other entry point.")
