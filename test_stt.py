"""
Speech-to-Text (STT) Test Script
Uses Whisper to transcribe your voice from microphone.
"""

import whisper
import numpy as np
import sounddevice as sd
import tempfile
import os
import sys
import wave
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 50)
print("WHISPER STT - SPEECH TO TEXT TEST")
print("=" * 50)

# Load Whisper model (tiny for fast testing)
print("\nLoading Whisper model (tiny)...")
model = whisper.load_model("tiny")
print("[OK] Model loaded successfully!")

# Audio recording settings
SAMPLE_RATE = 16000
DURATION = 5  # seconds

print(f"\nStarting recording in 3 seconds... Get ready to speak!")
import time
time.sleep(3)
print("Recording for 5 seconds... Speak now!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
print("[OK] Recording complete!")

# Save to temporary WAV file using wave module (no ffmpeg needed)
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    temp_path = f.name
    # Convert float32 to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(temp_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes (16-bit)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())

# Transcribe using Whisper's audio array directly
print("\nTranscribing...")
# Whisper can process numpy arrays directly
audio_float = audio.flatten().astype(np.float32)
result = model.transcribe(audio_float)
transcription = result["text"]

print("\n" + "=" * 50)
print("TRANSCRIPTION RESULT:")
print("=" * 50)
print(f"You said: \"{transcription.strip()}\"")
print("=" * 50)

# Cleanup
os.unlink(temp_path)

print("\n[OK] STT Test Complete!")
