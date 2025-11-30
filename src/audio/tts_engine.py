import os
import json
import re
import numpy as np
import onnxruntime as ort
import pyaudio
from unicodedata import normalize
from RealtimeTTS.engines.base_engine import BaseEngine

class SupertonicVoice:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __repr__(self):
        return f"{self.name}"

class SupertonicEngine(BaseEngine):
    def __init__(self, model_dir: str, voice_style_path: str, speed: float = 1.05, steps: int = 5, use_gpu: bool = False, volume: float = 1.0):
        """
        Initializes the Supertonic ONNX engine wrapped for RealtimeTTS.
        """
        super().__init__()
        self.model_name = "Supertonic"
        self.speed = speed
        self.steps = steps
        self.model_dir = model_dir
        self.volume = volume # Store volume factor
        
        # 1. Load Configuration
        cfg_path = os.path.join(model_dir, "tts.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"tts.json not found in {model_dir}")
        with open(cfg_path, "r") as f: 
            self.cfgs = json.load(f)

        # 2. Load Unicode Indexer
        indexer_path = os.path.join(model_dir, "unicode_indexer.json")
        with open(indexer_path, "r", encoding="utf-8") as f:
            self.indexer = json.load(f)

        # 3. Initialize ONNX Sessions
        providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.models = {}
        for name in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]:
            path = os.path.join(model_dir, f"{name}.onnx")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model {name}.onnx not found in {model_dir}")
            self.models[name] = ort.InferenceSession(path, sess_options=opts, providers=providers)

        # 4. Load Initial Voice
        self.set_voice(voice_style_path)

        # 5. Configs
        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        self.base_chunk_size = self.cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]

    def post_init(self):
        self.engine_name = "supertonic"

    def get_stream_info(self):
        """Returns PyAudio parameters: format, channels, rate"""
        return pyaudio.paInt16, 1, self.sample_rate

    def get_voices(self):
        """Scans the voice_styles directory for available JSONs"""
        voices = []
        # Assuming voice_styles is parallel to the onnx dir based on user structure
        # assets/supertonic/onnx -> assets/supertonic/voice_styles
        style_dir = os.path.abspath(os.path.join(self.model_dir, "..", "voice_styles"))
        
        if os.path.exists(style_dir):
            for f in os.listdir(style_dir):
                if f.endswith(".json"):
                    voices.append(SupertonicVoice(f.replace(".json", ""), os.path.join(style_dir, f)))
        return voices

    def set_voice(self, voice):
        """Sets the voice style tensors"""
        path = ""
        if isinstance(voice, SupertonicVoice):
            path = voice.path
        elif isinstance(voice, str):
            path = voice
            # If just a name is passed, try to find it
            if not os.path.exists(path):
                style_dir = os.path.abspath(os.path.join(self.model_dir, "..", "voice_styles"))
                possible_path = os.path.join(style_dir, f"{voice}.json")
                if os.path.exists(possible_path):
                    path = possible_path

        if not os.path.exists(path):
            print(f"Warning: Voice file {path} not found.")
            return

        with open(path, "r") as f:
            vs = json.load(f)
        
        ttl_dims = vs["style_ttl"]["dims"]
        dp_dims = vs["style_dp"]["dims"]
        
        # Batch size 1
        self.ttl_style = np.array(vs["style_ttl"]["data"], dtype=np.float32).reshape(1, ttl_dims[1], ttl_dims[2])
        self.dp_style = np.array(vs["style_dp"]["data"], dtype=np.float32).reshape(1, dp_dims[1], dp_dims[2])

    def set_voice_parameters(self, **voice_parameters):
        if "speed" in voice_parameters:
            self.speed = voice_parameters["speed"]
        if "steps" in voice_parameters:
            self.steps = int(voice_parameters["steps"])

    # --- Text Preprocessing (Internal) ---
    def _preprocess_text(self, text: str) -> str:
        text = normalize("NFKD", text)
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text) # Simple emoji removal
        replacements = {"–": "-", "‑": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'"}
        for k, v in replacements.items(): text = text.replace(k, v)
        text = re.sub(r"\s+", " ", text).strip()
        if not text: return "."
        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text): text += "."
        return text

    def _text_to_ids(self, text: str):
        text = self._preprocess_text(text)
        # Convert chars to unicode ordinals
        unicode_vals = [ord(c) for c in text]
        
        # Map via indexer (List lookup)
        mapped_vals = []
        for val in unicode_vals:
            if val < len(self.indexer):
                mapped_vals.append(self.indexer[val])
            else:
                mapped_vals.append(0)
        
        text_ids = np.array(mapped_vals, dtype=np.int64).reshape(1, -1)
        text_lengths = np.array([len(mapped_vals)], dtype=np.int64)
        
        # Create Mask
        mask = np.ones((1, 1, len(mapped_vals)), dtype=np.float32)
        return text_ids, mask

    def _sample_noisy_latent(self, duration: np.ndarray):
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        
        latent_len = max(1, ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32))
        latent_dim = self.ldim * self.chunk_compress_factor
        
        noisy_latent = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
        
        # Create Latent Mask
        latent_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len_actual = (wav_lengths + latent_size - 1) // latent_size
        
        ids = np.arange(0, latent_len)
        mask = (ids < np.expand_dims(latent_len_actual, axis=1)).astype(np.float32)
        latent_mask = mask.reshape(1, 1, latent_len)

        return noisy_latent * latent_mask, latent_mask

    def synthesize(self, text: str) -> bool:
        """
        The main entry point called by RealtimeTTS. 
        Generates audio and puts it into self.queue.
        """
        try:
            # 1. Prepare Text
            text_ids, text_mask = self._text_to_ids(text)
            
            # 2. Duration Predictor
            dur_onnx, *_ = self.models["duration_predictor"].run(None, {
                "text_ids": text_ids, "style_dp": self.dp_style, "text_mask": text_mask
            })
            dur_onnx = dur_onnx / self.speed
            
            # 3. Text Encoder
            text_emb_onnx, *_ = self.models["text_encoder"].run(None, {
                "text_ids": text_ids, "style_ttl": self.ttl_style, "text_mask": text_mask
            })
            
            # 4. Vector Estimation (Diffusion)
            xt, latent_mask = self._sample_noisy_latent(dur_onnx)
            total_step_np = np.array([self.steps], dtype=np.float32)
            
            for step in range(self.steps):
                current_step = np.array([step], dtype=np.float32)
                xt, *_ = self.models["vector_estimator"].run(None, {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": self.ttl_style,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                })
            
            # 5. Vocoder
            wav, *_ = self.models["vocoder"].run(None, {"latent": xt})
            
            # 6. Post Process
            valid_samples = int(self.sample_rate * dur_onnx[0].item())
            wav = wav[:, :valid_samples].flatten()
            
            # Apply volume scaling and clip to prevent overflow
            wav = np.clip(wav * self.volume, -1.0, 1.0)
            
            # Float32 to Int16
            audio_int16 = (wav * 32767).astype(np.int16).tobytes()
            
            # 7. Send to RealtimeTTS Queue
            self.queue.put(audio_int16)
            return True

        except Exception as e:
            print(f"Supertonic Synthesis Error: {e}")
            import traceback
            traceback.print_exc()
            return False