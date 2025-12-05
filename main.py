import argparse
import os
import sys
import json
import re
import time
from typing import Optional
from unicodedata import normalize

# Third-party imports
import numpy as np
import onnxruntime as ort
import pyaudio
import ollama
import colorama
from colorama import Fore, Style as ColorStyle
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from RealtimeSTT import AudioToTextRecorder

# ==========================================
# Supertonic TTS Logic (Embedded & Fixed)
# ==========================================

class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r", encoding="utf-8") as f:
            # This loads as a LIST, where index = unicode ordinal, value = token ID
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str) -> str:
        text = normalize("NFKD", text)
        
        # Remove emojis
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
            "\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        replacements = {
            "–": "-", "‑": "-", "—": "-", "¯": " ", "_": " ",
            "“": '"', "”": '"', "‘": "'", "’": "'", "´": "'", "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
            "→": " ", "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # Remove combining diacritics
        text = re.sub(r"[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030a\u030b\u030c\u0327\u0328\u0329\u032a\u032b\u032c\u032d\u032e\u032f]", "", text)
        text = re.sub(r"[♥☆♡©\\]", "", text)

        expr_replacements = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}
        for k, v in expr_replacements.items():
            text = text.replace(k, v)

        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)
        
        text = text.replace('""', '"').replace("''", "'").replace("``", "`")
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "."

        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."

        return text

    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        max_len = text_ids_lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(text_ids_lengths, axis=1)).astype(np.float32)
        return mask.reshape(-1, 1, max_len)

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        return np.array([ord(char) for char in text], dtype=np.uint16)

    def __call__(self, text_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        
        if len(text_list) == 0 or text_ids_lengths.max() == 0:
             return np.zeros((0,0), dtype=np.int64), np.zeros((0,0,0), dtype=np.float32)

        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            
            # FIX: Access self.indexer as a list using integer indices
            mapped_vals = []
            for val in unicode_vals:
                # Ensure the ordinal is within the bounds of the indexer list
                if val < len(self.indexer):
                    mapped_vals.append(self.indexer[val])
                else:
                    # Fallback for unknown characters (usually 0 is pad/unknown)
                    mapped_vals.append(0)
            
            text_ids[i, : len(unicode_vals)] = np.array(mapped_vals, dtype=np.int64)
        
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask

class Style:
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx

class TextToSpeech:
    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_ort: ort.InferenceSession,
        text_enc_ort: ort.InferenceSession,
        vector_est_ort: ort.InferenceSession,
        vocoder_ort: ort.InferenceSession,
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def _length_to_mask(self, lengths: np.ndarray) -> np.ndarray:
        max_len = lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
        return mask.reshape(-1, 1, max_len)

    def sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        
        latent_len = max(1, ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32))
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        
        latent_size = self.base_chunk_size * self.chunk_compress_factor
        latent_lengths = (wav_lengths + latent_size - 1) // latent_size
        latent_mask = self._length_to_mask(latent_lengths)
        
        if latent_mask.shape[-1] < latent_len:
            pad = np.zeros((bsz, 1, latent_len - latent_mask.shape[-1]), dtype=np.float32)
            latent_mask = np.concatenate([latent_mask, pad], axis=2)
        elif latent_mask.shape[-1] > latent_len:
            latent_mask = latent_mask[:, :, :latent_len]

        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(self, text_list: list[str], style: Style, total_step: int, speed: float) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list)
        
        dur_onnx, *_ = self.dp_ort.run(None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask})
        dur_onnx = dur_onnx / speed
        
        text_emb_onnx, *_ = self.text_enc_ort.run(None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask})
        
        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(None, {
                "noisy_latent": xt,
                "text_emb": text_emb_onnx,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step_np,
            })
        
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur_onnx

    def __call__(self, text: str, style: Style, total_step: int, speed: float = 1.05, silence_duration: float = 0.3) -> np.ndarray:
        chunks = self._chunk_text(text)
        wav_cat = None
        
        for chunk in chunks:
            if not chunk.strip(): continue
            try:
                wav, dur_onnx = self._infer([chunk], style, total_step, speed)
                
                # Trim silence/padding
                if wav.shape[1] > 0:
                    valid_samples = int(self.sample_rate * dur_onnx[0].item())
                    wav = wav[:, :valid_samples]

                if wav_cat is None:
                    wav_cat = wav
                else:
                    silence = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
                    wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
            except Exception as e:
                print(f"[Error in TTS inference]: {e}")
                import traceback
                traceback.print_exc()
        
        return wav_cat

    def _chunk_text(self, text: str, max_len: int = 300) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
        chunks = []
        for paragraph in paragraphs:
            if not paragraph: continue
            pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<=[.!?])\s+"
            sentences = re.split(pattern, paragraph)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_len:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk: chunks.append(current_chunk.strip())
                    current_chunk = sentence
            if current_chunk: chunks.append(current_chunk.strip())
        return chunks if chunks else [text]

def load_voice_style(voice_style_path: str) -> Style:
    with open(voice_style_path, "r") as f:
        voice_style = json.load(f)
    
    ttl_dims = voice_style["style_ttl"]["dims"]
    dp_dims = voice_style["style_dp"]["dims"]
    
    ttl_style = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).reshape(1, ttl_dims[1], ttl_dims[2])
    dp_style = np.array(voice_style["style_dp"]["data"], dtype=np.float32).reshape(1, dp_dims[1], dp_dims[2])
    
    return Style(ttl_style, dp_style)

def load_tts_system(onnx_dir: str, use_gpu: bool = False) -> TextToSpeech:
    providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    
    cfg_path = os.path.join(onnx_dir, "tts.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
        
    with open(cfg_path, "r") as f: cfgs = json.load(f)
    
    indexer_path = os.path.join(onnx_dir, "unicode_indexer.json")
    processor = UnicodeProcessor(indexer_path)
    
    models = {}
    for name in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]:
        path = os.path.join(onnx_dir, f"{name}.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        models[name] = ort.InferenceSession(path, sess_options=opts, providers=providers)
        
    return TextToSpeech(cfgs, processor, models["duration_predictor"], 
                       models["text_encoder"], models["vector_estimator"], models["vocoder"])

# ==========================================
# Main Application
# ==========================================

if __name__ == '__main__':
    
    # Fix for Windows PyTorch/Audio DLL paths
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        try:
            from torchaudio._extension.utils import _init_dll_path
            _init_dll_path()
        except ImportError:
            pass

    colorama.init()
    console = Console()

    parser = argparse.ArgumentParser(description='Start the realtime STT -> LLM -> Supertonic TTS assistant.')

    # STT Arguments
    parser.add_argument('-m', '--model', type=str, default='base',
                        help='Path to the STT model or model size. Default is base.')
    parser.add_argument('-r', '--rt-model', '--realtime_model_type', type=str, default='base',
                        help='Model size for real-time transcription. Default is base.')
    parser.add_argument('-l', '--lang', '--language', type=str, default='en',
                        help='Language code for the STT model. Default is en.')
    
    # LLM Arguments
    parser.add_argument('--ollama-model', type=str, default='gemma3:1b',
                        help='Ollama model to use for LLM responses. Default is gemma3:1b.')

    # Supertonic TTS Arguments
    parser.add_argument('--onnx-dir', type=str, default='assets/supertonic/onnx',
                        help='Path to the directory containing ONNX model files and configs.')
    parser.add_argument('--voice-style', type=str, default='assets/supertonic/voice_styles/F1.json',
                        help='Path to the voice style JSON file.')
    parser.add_argument('--speed', type=float, default=1.05,
                        help='Speech speed (higher = faster). Default 1.05.')
    parser.add_argument('--steps', type=int, default=5,
                        help='Number of denoising steps (higher = better quality, slower). Default 5.')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for TTS inference.')

    args = parser.parse_args()

    # Global State
    full_sentences = []
    conversation_history = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""
    is_processing = False
    is_speaking = False

    # Initialize Rich Live Display
    live = Live(console=console, refresh_per_second=10, screen=False, transient=True)
    live.start()

    # Load TTS System
    try:
        console.print("[yellow]Loading Supertonic TTS models...[/yellow]")
        tts_engine = load_tts_system(args.onnx_dir, args.use_gpu)
        voice_style_obj = load_voice_style(args.voice_style)
        console.print("[green]✓[/green] TTS initialized successfully")
    except Exception as e:
        live.stop()
        console.print(f"[red]✗[/red] Failed to initialize TTS: {e}")
        console.print("[yellow]Please ensure 'assets/onnx' and 'assets/voice_styles' exist.[/yellow]")
        sys.exit(1)

    # Personality System Prompt
    PERSONALITY = """You are Kiva, a playful, excited, and very cheerful voice assistant. You're energetic, 
    enthusiastic, and genuinely love helping people! 

    Your vibe:
    - Super playful and upbeat - you bring positive energy!
    - Excited about helping and solving problems
    - Friendly and warm, like a helpful buddy
    - Uses natural, conversational language
    - Quick with jokes and puns when appropriate

    Keep responses conversational and natural. Short, clear, and full of helpful energy. 
    Use contractions, be casual, and keep things light! 
    
    IMPORTANT: 
    - You're a helpful assistant, NOT a romantic interest
    - No emojis or asterisks in your output (e.g. no *laughs*)"""

    conversation_history.append({'role': 'system', 'content': PERSONALITY})

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0
    prev_text = ""

    def preprocess_text(text):
        text = text.lstrip()
        if text.startswith("..."): text = text[3:]
        text = text.lstrip()
        if text: text = text[0].upper() + text[1:]
        return text

    def text_to_speech_sync(text):
        """Convert text to speech using Supertonic and play via PyAudio"""
        global is_speaking
        try:
            is_speaking = True
            
            # Generate Audio (Returns float32 numpy array)
            wav_data = tts_engine(text, voice_style_obj, args.steps, args.speed)
            
            if wav_data is None or wav_data.size == 0:
                return

            # Flatten to 1D array
            wav_data = wav_data.flatten()
            
            # Convert float32 [-1, 1] to int16 [-32768, 32767] for PyAudio
            audio_data = (wav_data * 32767).astype(np.int16)
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=tts_engine.sample_rate,
                output=True,
                frames_per_buffer=4096
            )
            
            # Write data
            stream.write(audio_data.tobytes())
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            # We avoid printing here to not break the Live display layout
            pass
        finally:
            is_speaking = False

    def get_llm_response(user_text, model_name):
        try:
            conversation_history.append({'role': 'user', 'content': user_text})
            
            response = ollama.chat(
                model=model_name,
                messages=conversation_history[-21:], 
                stream=False
            )
            
            assistant_response = response['message']['content']
            conversation_history.append({'role': 'assistant', 'content': assistant_response})
            return assistant_response
            
        except Exception as e:
            return "Oops, something went wrong on my end! Can you say that again?"

    def print_conversation_item(role, text):
        """Prints a permanent message to the console, temporarily stopping Live display."""
        live.stop()
        if role == "user":
            console.print(Panel(Text(text, style="cyan"), title="[bold cyan]You[/bold cyan]", border_style="cyan", expand=False))
        else:
            console.print(Panel(Text(text, style="magenta"), title="[bold magenta]Kiva[/bold magenta]", border_style="magenta", expand=False))
        live.start()

    def update_status(text, title="Kiva Voice Assistant", style="bold yellow"):
        """Updates the sticky bottom status panel."""
        panel = Panel(Text(text, style="yellow", justify="center"), title=title, border_style=style)
        live.update(panel)

    def text_detected(text):
        global prev_text
        if is_processing or is_speaking: return

        text = preprocess_text(text)
        sentence_end_marks = ['.', '!', '?', '。'] 
        
        if text.endswith("..."):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        if text:
            update_status(f"Listening: {text}", title="[bold green]🎤 Listening...[/bold green]", style="bold green")

    def process_text(text):
        global recorder, full_sentences, prev_text, is_processing
        
        if is_processing: return
        is_processing = True
        
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause
        text = preprocess_text(text).rstrip()
        if text.endswith("..."): text = text[:-2]
        
        if not text:
            is_processing = False
            return

        # Print User Message
        full_sentences.append(text)
        prev_text = ""
        print_conversation_item("user", text)
        
        # Thinking State
        update_status("💭 Kiva is thinking...", title="[bold yellow]Processing...[/bold yellow]", style="bold yellow")
        
        # LLM Generation
        llm_response = get_llm_response(text, args.ollama_model)
        full_sentences.append(llm_response)
        
        # Print Assistant Message
        print_conversation_item("kiva", llm_response)
        
        # Speaking State
        update_status("🔊 Kiva is speaking...", title="[bold blue]Speaking...[/bold blue]", style="bold blue")
        
        # TTS Playback
        text_to_speech_sync(llm_response)
        
        # Ready State
        update_status("Ready for your response!", title="[bold green]🎤 Listening...[/bold green]", style="bold green")
        
        is_processing = False

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': args.model,
        'realtime_model_type': args.rt_model,
        'language': args.lang,
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'min_length_of_recording': 1.1,        
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
        'early_transcription_on_silence': 0,
        'beam_size': 5,
        'beam_size_realtime': 3,
        'batch_size': 2,
        'realtime_batch_size': 2,        
        'no_log_file': True,
        'initial_prompt_realtime': "End incomplete sentences with ellipses...",
        'silero_use_onnx': True,
        'faster_whisper_vad_filter': False,
    }

    recorder = AudioToTextRecorder(**recorder_config)
    
    initial_text = Panel(
        Text("Hey there! I'm Kiva, your friendly voice assistant!\n\nStart talking whenever you're ready!\nI'm excited to help you out!\n\n(Press Ctrl+C to exit)", 
             style="cyan bold", justify="center"), 
        title="[bold yellow]Kiva Voice Assistant[/bold yellow]", 
        border_style="bold yellow"
    )
    live.update(initial_text)

    console.print(f"\n[bold magenta]Kiva is using:[/bold magenta] {args.ollama_model}")
    console.print(f"[bold magenta]Voice Style:[/bold magenta] {os.path.basename(args.voice_style)}")
    console.print(f"[bold cyan]Audio Engine:[/bold cyan] PyAudio")
    console.print(f"[bold yellow]Tip: Speak naturally and wait for Kiva to finish speaking before you respond![/bold yellow]\n")

    try:
        while True:
            if not is_processing and not is_speaking:
                recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("\n[bold green]Thanks for chatting! See you next time![/bold green]")
        console.print("[bold yellow]- Kiva[/bold yellow]\n")
        exit(0)