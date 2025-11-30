import os
import sys
import warnings
import time

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. SUPPRESS NOISE ---
warnings.filterwarnings("ignore")
from src.core.suppress import ignore_stderr, configure_logging, suppress_pydub_warnings

suppress_pydub_warnings()
configure_logging()

# --- 3. IMPORTS ---
from src.core.config import load_config
from src.ui.tui import KivaTUI
from src.llm.client import OllamaClient
# Defer heavy imports to inside try/except or after UI init if possible, 
# but for now we import them here to ensure they load correctly.

def main():
    # Initialize TUI (but don't start Live yet if we want to log startup)
    # Actually, let's use TUI for everything once it starts.
    tui = KivaTUI()
    
    # We can use console.status for startup
    with tui.console.status("[bold green]System initializing...") as status:
        
        # --- Config Load ---
        try:
            status.update("[bold cyan]Loading configuration...")
            cfg = load_config()
        except Exception as e:
            tui.console.print(f"[red]Config Error:[/red] {e}")
            return

        # --- LLM Initialization ---
        status.update(f"[bold cyan]Connecting to Ollama ({cfg['llm']['model']})...")
        llm_client = OllamaClient(
            model_name=cfg['llm']['model'],
            system_prompt=cfg['llm']['system_prompt']
        )

        # State for callback
        is_speaking = False

        def on_audio_start():
            nonlocal is_speaking
            is_speaking = True
            tui.set_status("Speaking...")

        def on_audio_stop():
            nonlocal is_speaking
            is_speaking = False
            tui.set_status("Listening...")

        # --- TTS Initialization ---
        status.update("[bold cyan]Loading Supertonic TTS Model...")
        with ignore_stderr():
            from RealtimeTTS import TextToAudioStream
            from src.audio.tts_engine import SupertonicEngine
            
            tts_engine = SupertonicEngine(
                model_dir=cfg['tts']['model_dir'],
                voice_style_path=cfg['tts']['voice_style_path'],
                speed=cfg['tts']['speed'],
                steps=cfg['tts']['steps'],
                use_gpu=cfg['tts']['use_gpu'],
                volume=cfg['tts']['volume']
            )
            tts_stream = TextToAudioStream(
                tts_engine, 
                log_characters=False,
                on_audio_stream_start=on_audio_start,
                on_audio_stream_stop=on_audio_stop
            )
        
        tui.console.print("[green]✓[/green] TTS Engine Ready")

        # --- STT Initialization ---
        status.update("[bold cyan]Loading RealtimeSTT (Whisper)...")
        
        def text_detected_callback(text):
            if not is_speaking:
                tui.set_realtime_text(text)
                tui.set_status("Listening...")

        with ignore_stderr():
            from RealtimeSTT import AudioToTextRecorder
            
            recorder = AudioToTextRecorder(
                model=cfg['stt']['model'],
                realtime_model_type=cfg['stt']['realtime_model'],
                language=cfg['stt']['language'],
                spinner=False, 
                silero_sensitivity=0.4,
                post_speech_silence_duration=0.6,
                on_realtime_transcription_update=text_detected_callback,
                no_log_file=True
            )
        
        tui.console.print("[green]✓[/green] STT Engine Ready")

    # --- Start Main Loop ---
    tui.start()
    tui.set_status("Listening...")
    tui.add_assistant_message("Hey! I'm Kiva. I'm ready to chat!")

    def process_text(text):
        nonlocal is_speaking
        text = text.strip()
        if not text: return

        # 1. User Input
        tui.add_user_message(text) 
        tui.set_status("Processing...")
        
        # 2. Streaming Pipeline
        is_speaking = True 
        
        try:
            # Get generator
            stream = llm_client.chat(text, stream=True)
            
            def write_chunks(stream_gen):
                first_chunk = True
                for chunk in stream_gen:
                    if not chunk: continue
                    
                    # Update UI
                    if first_chunk:
                        # Start new bubble
                        tui.add_assistant_message(chunk)
                        first_chunk = False
                    else:
                        # Append to bubble
                        tui.update_last_assistant_message(chunk)
                    
                    yield chunk

            # Feed TTS with generator that updates UI as side effect
            tts_stream.feed(write_chunks(stream))
            
            if not tts_stream.is_playing():
                tts_stream.play_async(
                    fast_sentence_fragment=True, 
                    buffer_threshold_seconds=1.0, 
                    log_synthesized_text=False
                )
                
        except Exception as e:
            tui.console.print(f"[red]LLM Error: {e}[/red]")
            is_speaking = False # Reset if error

    try:
        while True:
            # Wait if audio is playing or we are conceptually "speaking"
            if is_speaking or tts_stream.is_playing():
                time.sleep(0.1)
            else:
                tui.set_status("Listening...")
                user_text = recorder.text()
                if user_text:
                    process_text(user_text)

    except KeyboardInterrupt:
        tui.stop()
        print("\n\nKiva: Goodbye! See you next time!")
        try:
            recorder.shutdown()
            tts_stream.stop()
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()
