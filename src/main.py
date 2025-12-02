import os
import sys
import warnings
import time
import threading
from pathlib import Path

# --- 1. SETUP PATHS ---
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- 2. SUPPRESS NOISE ---
warnings.filterwarnings("ignore")
from src.core.suppress import ignore_stderr, configure_logging, suppress_pydub_warnings

suppress_pydub_warnings()
configure_logging()

# --- 3. IMPORTS ---
from src.core.config import load_config
from src.ui.tui import KivaTUI
from src.llm.client import OllamaClient
from src.audio.wakeword import WakewordDetector
from src.audio.stt_service import STTService

def main():
    # Initialize TUI
    tui = KivaTUI()
    tui.start() # Start Live display immediately to capture startup logs in UI if we wanted
    
    # Use a temporary status update since we can't use console.status context manager easily with Live active
    tui.set_status("System initializing...")

    # --- Config Load ---
    try:
        tui.set_status("Loading configuration...")
        cfg = load_config()
    except Exception as e:
        tui.add_system_message(f"Config Error: {e}")
        return

    # --- LLM Initialization ---
    tui.set_status(f"Connecting to Ollama ({cfg['llm']['model']})...")
    try:
        llm_client = OllamaClient(
            model_name=cfg['llm']['model'],
            system_prompt=cfg['llm']['system_prompt']
        )
        tui.add_system_message(f"LLM Client Ready: {cfg['llm']['model']}")
    except Exception as e:
        tui.add_system_message(f"LLM Error: {e}")
        return

    # Thread-safe event to track if Kiva is currently speaking
    speaking_event = threading.Event()

    # --- TTS Initialization ---
    tui.set_status("Loading Supertonic TTS Model...")
    with ignore_stderr():
        from RealtimeTTS import TextToAudioStream
        from src.audio.tts_engine import SupertonicEngine
        
        def on_speak_start():
            speaking_event.set()
            tui.set_status("Speaking...")
            
        def on_speak_stop():
            speaking_event.clear()
            tui.set_status("Waiting for Wake Word...")

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
            on_audio_stream_start=on_speak_start,
            on_audio_stream_stop=on_speak_stop
        )
    tui.add_system_message("TTS Engine Ready")

    # --- Wakeword Initialization ---
    tui.set_status("Loading Wakeword Engine...")
    
    wakeword_paths = [
        str(project_root / path) 
        for path in cfg['wakewords']['model_paths']
    ]
    
    wakeword_detector = WakewordDetector(
        model_paths=wakeword_paths,
        sensitivity=cfg['wakewords']['sensitivity']
    )
    tui.add_system_message(f"Wakewords Loaded: {len(wakeword_paths)}")


    # --- STT Initialization ---
    tui.set_status("Loading RealtimeSTT (Whisper)...")
    
    def text_detected_callback(text):
        # Only update if Kiva is not speaking
        if not speaking_event.is_set():
            tui.set_realtime_text(text)
            tui.set_status("Listening...")

    with ignore_stderr():
        stt_service = STTService(
            model=cfg['stt']['model'],
            realtime_model=cfg['stt']['realtime_model'],
            language=cfg['stt']['language'],
            silero_sensitivity=0.4,
            post_speech_silence_duration=0.6,
            on_realtime_transcription_update=text_detected_callback
        )
    tui.add_system_message("STT Engine Ready")


    tui.set_status("Waiting for Wake Word...")
    tui.add_assistant_message("Hey! I'm Kiva. I'm ready to chat!")

    def process_text(text: str):
        text = text.strip()
        if not text: return

        # 1. User Input
        tui.add_user_message(text) 
        tui.set_status("Processing...")
        
        # 2. Streaming Pipeline
        # speaking_event managed by TTS callbacks
        
        # Create an empty message bubble for the assistant's response
        tui.add_assistant_message("") 
        
        def processing_llm_stream(llm_output_stream):
            """
            Processes the LLM output stream:
            - Updates the TUI with raw content (including markdown).
            - Filters out code blocks (```...```) and asterisks (*) for the TTS engine.
            """
            in_code_block = False
            buffer = "" 
            
            for raw_chunk in llm_output_stream:
                # 1. Update TUI with RAW content immediately
                tui.update_last_assistant_message(raw_chunk)
                
                # 2. Process for TTS
                buffer += raw_chunk
                
                while True:
                    if not in_code_block:
                        code_start_idx = buffer.find("```")
                        
                        if code_start_idx != -1:
                            to_yield = buffer[:code_start_idx].replace("*", "")
                            if to_yield: yield to_yield
                            in_code_block = True
                            buffer = buffer[code_start_idx + 3:] 
                        else:
                            to_yield = buffer.replace("*", "")
                            if to_yield: yield to_yield
                            buffer = "" 
                            break 
                    else: 
                        code_end_idx = buffer.find("```")
                        
                        if code_end_idx != -1:
                            in_code_block = False
                            buffer = buffer[code_end_idx + 3:] 
                        else:
                            buffer = "" 
                            break 
            
            if buffer and not in_code_block:
                to_yield = buffer.replace("*", "")
                if to_yield: yield to_yield

        try:
            # Get generator (LLM output stream)
            stream = llm_client.chat(text, stream=True)
            
            # Feed the processed stream to RealtimeTTS
            tts_stream.feed(processing_llm_stream(stream))
            
            # Start playing if not already playing
            if not tts_stream.is_playing():
                tts_stream.play_async(
                    fast_sentence_fragment=True, 
                    buffer_threshold_seconds=1.0, 
                    log_synthesized_text=False
                )
                
        except Exception as e:
            tui.add_system_message(f"LLM Error: {e}")

    try:
        while True:
            # Wait if audio is playing or we are conceptually "speaking"
            if speaking_event.is_set() or tts_stream.is_playing():
                time.sleep(0.1)
            else:
                tui.set_status("Waiting for Wake Word...")
                
                # 1. Wait for Wakeword
                wakeword_detected = False
                def on_wake():
                    nonlocal wakeword_detected
                    wakeword_detected = True
                
                # This blocks until wakeword detected
                # We need to run this in a way that doesn't block the TUI live update?
                # Actually, TUI.Live runs in a separate thread by default in rich? No, it updates on .update().
                # If we block here, TUI won't refresh if refresh_per_second is driven by main thread?
                # Rich Live uses a thread for refreshing if refresh_per_second is set.
                # So blocking is fine for the display, but we must ensure we don't block forever if we want to handle exit.
                wakeword_detector.start(on_detected=on_wake)
                
                if wakeword_detected:
                    tui.set_status("Wake Word Detected! Listening...")
                    
                    # 2. Listen for Command (VAD controlled)
                    user_text = stt_service.get_text()
                    if user_text:
                        process_text(user_text)

    except KeyboardInterrupt:
        tui.stop()
        print("\n\nKiva: Goodbye! See you next time!")
        try:
            wakeword_detector.stop()
            wakeword_detector.cleanup()
            stt_service.shutdown()
            tts_stream.stop()
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()