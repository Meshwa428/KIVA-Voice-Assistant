# RealtimeTTS wrappers

from RealtimeTTS import TextToAudioStream, SystemEngine

class TTSClient:
    """
    A client for the RealtimeTTS library.
    """
    def __init__(self, engine=None, on_character=None, on_word=None, **kwargs):
        """
        Initializes the TTSClient.

        Args:
            engine: The TTS engine to use. If None, defaults to SystemEngine.
            on_character (callable, optional): Callback for each character processed.
            on_word (callable, optional): Callback for each word played.
            **kwargs: Additional arguments for TextToAudioStream.
        """
        if engine is None:
            self.engine = SystemEngine()
        else:
            self.engine = engine

        self.stream = TextToAudioStream(
            self.engine,
            on_character=on_character,
            on_word=on_word,
            **kwargs
        )
        print("TTS client initialized.")

    def feed(self, text):
        """
        Feeds text to the TTS stream.

        Args:
            text (str or generator): The text to be synthesized.
        """
        self.stream.feed(text)

    def play(self, output_wavfile=None, **kwargs):
        """
        Plays the synthesized audio.

        Args:
            output_wavfile (str, optional): If provided, the audio will be saved to this file.
            **kwargs: Additional arguments for the play method.
        """
        print("Playing audio...")
        self.stream.play(output_wavfile=output_wavfile, **kwargs)

    def play_async(self, output_wavfile=None, **kwargs):
        """
        Plays the synthesized audio asynchronously.

        Args:
            output_wavfile (str, optional): If provided, the audio will be saved to this file.
            **kwargs: Additional arguments for the play_async method.
        """
        print("Playing audio asynchronously...")
        self.stream.play_async(output_wavfile=output_wavfile, **kwargs)

    def is_playing(self):
        """
        Checks if the audio is currently playing.
        """
        return self.stream.is_playing()

    def stop(self):
        """
        Stops the audio stream immediately.
        """
        self.stream.stop()
