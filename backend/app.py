# FastAPI gateway, MCP adapters

import asyncio
import ollama
import threading
from audio.stt_client import STTClient
from audio.tts_client import TTSClient
from backend.agents.voice_agent import create_voice_agent
from functools import partial

class VoiceAssistant:
    """
    The main voice assistant class that orchestrates STT, LLM, and TTS.
    """
    def __init__(self, model: str = "llama2", on_final_transcript=None, on_partial_transcript=None, on_llm_token=None, request_approval=None):
        """
        Initializes the VoiceAssistant.
        """
        self.stt_client = STTClient(
            on_final_transcript=self._on_final_transcript,
            on_partial_transcript=self._on_partial_transcript
        )
        self.tts_client = TTSClient(on_character=self.handle_llm_token)
        self.agent_executor = create_voice_agent()
        self.model = model
        self.loop = asyncio.get_event_loop()
        self.stt_thread = None
        self.on_final_transcript = on_final_transcript
        self.on_partial_transcript = on_partial_transcript
        self.on_llm_token = on_llm_token
        self.request_approval = request_approval
        print("Voice Assistant with LangChain agent initialized.")

    def handle_llm_token(self, token: str):
        """
        Handles an LLM token from the TTS client.
        """
        if self.on_llm_token:
            self.on_llm_token(token)

    def _on_partial_transcript(self, text: str):
        """
        Thread-safe callback for when a partial transcript is received.
        """
        if self.on_partial_transcript:
            self.on_partial_transcript(text)

    async def handle_transcript(self, text: str):
        """
        Handles the transcribed text from the STT client.
        This function is called as a callback from a separate thread.
        """
        if self.on_final_transcript:
            self.on_final_transcript(text)

        if text.lower().startswith(("hey assistant", "assistant")):
            command = text.lower().replace("hey assistant", "").replace("assistant", "").strip()
            print(f"Agent command: {command}")

            # Simple check for destructive command
            if "write_file" in command:
                if self.request_approval:
                    approved = await self.request_approval(f"Do you want to execute this command: '{command}'?")
                    if not approved:
                        self.tts_client.feed("Command cancelled.")
                        self.tts_client.play_async()
                        return

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, partial(self.agent_executor.invoke, {"input": command})
            )

            agent_response = result.get("output", "I'm not sure how to respond to that.")
            print(f"Agent response: {agent_response}")
            self.tts_client.feed(agent_response)
            self.tts_client.play_async()
            return

        print("Sending to Ollama for simple chat...")

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
        Schedules the async handler on the main event loop.
        """
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.handle_transcript(text), self.loop)
        else:
            print("Event loop not running, cannot handle transcript.")

    def run(self):
        """
        Starts the voice assistant.
        """
        print("Starting Voice Assistant...")

        self.stt_thread = threading.Thread(target=self.stt_client.start)
        self.stt_thread.daemon = True
        self.stt_thread.start()

        print("Voice Assistant is running. Press Ctrl+C to exit.")
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            print("Shutting down Voice Assistant.")
        finally:
            self.loop.close()
            # Note: The STT thread is a daemon, so it will exit when the main thread exits.
            # A more graceful shutdown would involve signaling the thread to stop.


if __name__ == '__main__':
    # This script is intended to be run as part of a larger application
    # with a running asyncio event loop (e.g., a Textual TUI or FastAPI server).
    # For standalone testing, you can run it like this:

    # assistant = VoiceAssistant()
    # assistant.run()

    print("Backend app structure created. This will be orchestrated by a TUI or other entry point.")
