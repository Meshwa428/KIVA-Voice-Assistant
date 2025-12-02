import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from typing import List, Callable, Optional
import time

class WakewordDetector:
    def __init__(
        self, 
        model_paths: List[str], 
        sensitivity: float = 0.5,
        chunk_size: int = 1280, # 1280 samples = 80ms at 16kHz
        device_index: Optional[int] = None
    ):
        self.model_paths = model_paths
        self.sensitivity = sensitivity
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.running = False
        
        # Initialize OpenWakeWord Model
        # inference_framework="onnx" is default
        self.model = Model(wakeword_model_paths=model_paths)
        
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start(self, on_detected: Callable[[], None]):
        """
        Starts listening for the wakeword in a blocking loop (or separate thread if you wrap it).
        For this synchronous implementation, it blocks until wakeword is detected.
        """
        self.running = True
        
        # Open stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index
        )
        
        print(f"Listening for wakewords: {self.model_paths}")
        
        try:
            while self.running:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Feed to model
                # predict() expects 1D numpy array of int16 or float32
                prediction = self.model.predict(audio_data)
                
                # Check scores
                # prediction is dict: {'model_name': score, ...}
                for mdl, score in prediction.items():
                    if score > self.sensitivity:
                        self.running = False # Stop listening
                        on_detected()
                        return
                        
        except KeyboardInterrupt:
            self.stop()
            raise
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def cleanup(self):
        self.p.terminate()
