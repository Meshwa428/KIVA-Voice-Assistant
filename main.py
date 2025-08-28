import asyncio
import ollama
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, SystemEngine
import time

def transcribe_audio(recorder, audio_file):
    """
    Transcribes the audio from the given file.
    """
    print(f"[{time.time()}] Starting transcription for {audio_file}...")
    try:
        with open(audio_file, "rb") as f:
            audio_chunk = f.read()
        print(f"[{time.time()}] Feeding audio chunk to recorder...")
        recorder.feed_audio(audio_chunk)
        print(f"[{time.time()}] Waiting for transcription result...")
        text = recorder.text()
        print(f"[{time.time()}] Transcription finished.")
        print(f"Transcription: {text}")
        return text
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file}")
        return None

async def main():
    print(f"[{time.time()}] Starting PoC script...")
    audio_file = "test_audio.pcm"

    print(f"[{time.time()}] Initializing RealtimeSTT...")
    stt_recorder = AudioToTextRecorder(use_microphone=False, model="tiny.en")
    print(f"[{time.time()}] RealtimeSTT initialized.")

    print(f"[{time.time()}] Calling transcribe_audio...")
    transcribed_text = transcribe_audio(stt_recorder, audio_file)
    print(f"[{time.time()}] transcribe_audio returned.")

    if transcribed_text is not None:
        print(f"[{time.time()}] Sending text to Ollama...")
        response = ollama.chat(model='llama2', messages=[
            {
                'role': 'user',
                'content': transcribed_text,
            },
        ])
        ollama_response = response['message']['content']
        print(f"[{time.time()}] Ollama response received.")
        print(f"Ollama response: {ollama_response}")

        print(f"[{time.time()}] Synthesizing response with RealtimeTTS...")
        engine = SystemEngine()
        stream = TextToAudioStream(engine)
        stream.feed(ollama_response)
        stream.play(output_wavfile="response.wav")
        print(f"[{time.time()}] Response saved to response.wav")
    else:
        print(f"[{time.time()}] Transcription failed or produced no text.")


if __name__ == "__main__":
    asyncio.run(main())
