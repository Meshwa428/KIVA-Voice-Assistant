# RealtimeSTT wrappers

from RealtimeSTT import AudioToTextRecorder

class STTClient:
    """
    A client for the RealtimeSTT library that supports real-time transcription.
    """
    def __init__(self, on_final_transcript=None, on_partial_transcript=None, **kwargs):
        """
        Initializes the STTClient.

        Args:
            on_final_transcript (callable): Callback for when a final transcript is ready.
            on_partial_transcript (callable): Callback for real-time partial transcripts.
            **kwargs: Additional arguments for AudioToTextRecorder.
        """
        self.on_final_transcript = on_final_transcript

        self.recorder = AudioToTextRecorder(
            on_realtime_transcription_update=on_partial_transcript,
            enable_realtime_transcription=True,
            **kwargs
        )
        print("STT client initialized for real-time transcription.")

    def start(self):
        """
        Starts the STT client.
        The final transcript callback is passed to the recorder here.
        """
        print("STT client started. Listening for speech...")
        try:
            while True:
                if self.on_final_transcript:
                    self.recorder.text(self.on_final_transcript)
        except KeyboardInterrupt:
            print("STT client stopped by user.")

    def stop(self):
        """
        Stops the STT client.
        """
        # The loop in start() must be broken to stop this.
        print("Stopping STT client...")
        pass
