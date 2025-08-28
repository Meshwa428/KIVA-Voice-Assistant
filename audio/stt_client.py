# RealtimeSTT wrappers

from RealtimeSTT import AudioToTextRecorder

class STTClient:
    """
    A client for the RealtimeSTT library.
    """
    def __init__(self, model="tiny.en", language="en", **kwargs):
        """
        Initializes the STTClient.

        Args:
            model (str): The Whisper model to use for transcription.
            language (str): The language for transcription.
            **kwargs: Additional arguments for AudioToTextRecorder.
        """
        self.recorder = AudioToTextRecorder(model=model, language=language, **kwargs)
        print("STT client initialized.")

    def start(self, on_transcript):
        """
        Starts the STT client and registers a callback for transcriptions.

        This method runs in a loop to continuously listen for and process speech.

        Args:
            on_transcript (callable): A function to be called when a transcript is available.
                                      It should accept one argument: the transcript text (str).
        """
        print("STT client started. Listening for speech...")
        try:
            while True:
                self.recorder.text(on_transcript)
        except KeyboardInterrupt:
            print("STT client stopped by user.")

    def stop(self):
        """
        Stops the STT client.
        """
        # The loop in start() must be broken to stop this.
        # In a real application, we would use threading events or similar.
        print("Stopping STT client...")
        # The recorder itself doesn't have a persistent stop method for this pattern,
        # so we rely on breaking the loop in start().
        pass
