import wave
import struct
import math

def generate_wav_with_tone(filename="test_audio.wav", duration=1, sample_rate=16000, freq=440):
    """
    Generates a WAV file with a sine wave tone.
    """
    n_samples = int(duration * sample_rate)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            # Generate a sine wave sample
            sample = 32767 * math.sin(2 * math.pi * freq * i / sample_rate)
            wav_file.writeframes(struct.pack('<h', int(sample)))
    print(f"Generated WAV file with tone: {filename}")

def convert_wav_to_pcm(wav_filename="test_audio.wav", pcm_filename="test_audio.pcm"):
    """
    Converts a WAV file to a raw PCM file.
    """
    with wave.open(wav_filename, 'rb') as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != 16000:
            raise ValueError("WAV file must be 16-bit mono 16000Hz")

        with open(pcm_filename, 'wb') as pcm_file:
            pcm_file.write(wav_file.readframes(wav_file.getnframes()))
    print(f"Converted {wav_filename} to {pcm_filename}")

if __name__ == "__main__":
    generate_wav_with_tone()
    convert_wav_to_pcm()
