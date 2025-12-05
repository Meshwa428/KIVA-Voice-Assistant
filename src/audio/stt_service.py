from RealtimeSTT import AudioToTextRecorder
from typing import Callable, Optional

class STTService:
    def __init__(
        self, 
        model: str = "base",
        realtime_model: str = "base",
        language: str = "en",
        silero_sensitivity: float = 0.4,
        post_speech_silence_duration: float = 0.6,
        on_realtime_transcription_update: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the STT Service wrapper around RealtimeSTT.
        """
        self.recorder = AudioToTextRecorder(
            model=model,
            realtime_model_type=realtime_model,
            language=language,
            # VAD Config
            spinner=False, 
            silero_sensitivity=silero_sensitivity,
            post_speech_silence_duration=post_speech_silence_duration,
            on_realtime_transcription_update=on_realtime_transcription_update,
            enable_realtime_transcription=True,
            realtime_processing_pause=0.02,
            no_log_file=True
        )

    def get_text(self) -> str:
        """
        Blocking call that waits for speech and returns the transcribed text.
        """
        return self.recorder.text()

    def shutdown(self):
        """
        Clean up resources.
        """
        try:
            self.recorder.shutdown()
        except Exception:
            pass
