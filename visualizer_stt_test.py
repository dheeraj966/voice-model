"""
Combined Audio Visualizer + STT Test
Shows real-time visualization while recording, then transcribes.
"""

import numpy as np
import sounddevice as sd
import whisper
import sys
import os
import time
import wave
import tempfile
from collections import deque

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Settings
SAMPLE_RATE = 16000  # Whisper uses 16kHz
FFT_SIZE = 1024
BAR_COUNT = 16
RECORD_DURATION = 5  # seconds

print("=" * 60)
print("AUDIO VISUALIZER + STT TRANSCRIPTION TEST")
print("=" * 60)
print("\nThis will:")
print("1. Show real-time audio visualization")
print("2. Record your voice for 5 seconds")
print("3. Transcribe what you said using Whisper")
print("\nSpeak naturally when you see [RECORDING]!")
print("=" * 60)
print("\nStarting in 3 seconds...")
time.sleep(3)

# Load Whisper model first
print("\nLoading Whisper model (tiny)...")
model = whisper.load_model("tiny")
print("[OK] Model loaded!")
time.sleep(1)

# Audio buffer for visualization and recording
audio_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))  # 100ms for viz
recorded_audio = []
recording = False
recording_start_time = None

def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream."""
    global recording, recorded_audio, recording_start_time
    
    if recording:
        recorded_audio.append(indata.copy())
        if len(recorded_audio) >= int(SAMPLE_RATE * RECORD_DURATION / len(indata)):
            recording = False
    
    audio_buffer.extend(indata.copy())

def calculate_bars(audio_data):
    """Calculate frequency bars from audio data."""
    if len(audio_data) < FFT_SIZE:
        return [0] * BAR_COUNT
    
    window = np.hanning(len(audio_data))
    windowed = audio_data * window
    fft = np.abs(np.fft.rfft(windowed))
    
    bars = []
    freq_per_bar = len(fft) // BAR_COUNT
    for i in range(BAR_COUNT):
        start = i * freq_per_bar
        end = start + freq_per_bar
        if end <= len(fft):
            bar_value = np.mean(fft[start:end])
            bars.append(bar_value)
    
    max_val = max(bars) if max(bars) > 0 else 1
    bars = [min(1.0, v / max_val) for v in bars]
    
    return bars

def draw_visualizer(bars, volume, status_msg=""):
    """Draw the equalizer in terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 60)
    print("AUDIO VISUALIZER + STT")
    print("=" * 60)
    print()
    
    # Status
    if status_msg:
        print(f"  {status_msg}")
        print()
    
    # Volume meter
    vol_percent = int(volume * 100)
    vol_bar = "█" * int(volume * 40)
    vol_empty = "░" * (40 - len(vol_bar))
    print(f"VOLUME: [{vol_bar}{vol_empty}] {vol_percent:3d}%")
    print()
    
    # Frequency bars
    print("FREQUENCY SPECTRUM:")
    print("-" * 60)
    
    bar_height = 10
    for row in range(bar_height, 0, -1):
        line = "  "
        for bar in bars:
            bar_fill = int(bar * bar_height)
            if bar_fill >= row:
                line += "██ "
            elif bar_fill > row - 0.5:
                line += "▄▄ "
            else:
                line += "   "
        print(line)
    
    print("  " + "─" * (BAR_COUNT * 3))
    print("   LOW                    MID                   HIGH")
    print()
    
    # Level indicator
    if volume > 0.7:
        level = "LOUD 🔊"
    elif volume > 0.3:
        level = "MODERATE 🔉"
    elif volume > 0.05:
        level = "QUIET 🔈"
    else:
        level = "SILENCE 🔇"
    
    print(f"Audio Level: {level}")
    print("=" * 60)

# Start audio stream
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=FFT_SIZE
)
stream.start()

print("\nStarting visualization... Speak when ready!")
time.sleep(2)

# Visualization phase
viz_countdown = 5
while viz_countdown > 0:
    if len(audio_buffer) >= FFT_SIZE:
        audio_data = np.array(audio_buffer)[-FFT_SIZE:]
        rms = np.sqrt(np.mean(audio_data ** 2))
        volume = min(1.0, rms / 0.1)
        bars = calculate_bars(audio_data)
        draw_visualizer(bars, volume, f"Speak in {viz_countdown} seconds...")
    time.sleep(0.1)
    viz_countdown -= 1

# Recording phase
print("\n" * 2)
recording = True
recorded_audio = []
print("🔴 RECORDING NOW - SPEAK FOR 5 SECONDS!")
time.sleep(0.5)

record_start = time.time()
while recording and (time.time() - record_start) < RECORD_DURATION:
    if len(audio_buffer) >= FFT_SIZE:
        audio_data = np.array(audio_buffer)[-FFT_SIZE:]
        rms = np.sqrt(np.mean(audio_data ** 2))
        volume = min(1.0, rms / 0.1)
        bars = calculate_bars(audio_data)
        
        elapsed = int(time.time() - record_start)
        remaining = RECORD_DURATION - elapsed
        draw_visualizer(bars, volume, f"🔴 RECORDING... {remaining}s remaining - KEEP SPEAKING!")
    
    time.sleep(0.05)

# Stop recording
recording = False
stream.stop()
stream.close()

# Combine recorded audio
if len(recorded_audio) > 0:
    audio_concat = np.concatenate(recorded_audio, axis=0)
    audio_flat = audio_concat.flatten().astype(np.float32)
    
    # Save to file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        audio_int16 = (audio_concat * 32767).astype(np.int16)
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
    
    # Transcribe
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print()
    print("Transcribing your speech...")
    print()
    
    result = model.transcribe(audio_flat)
    transcription = result["text"]
    
    print("-" * 60)
    print("YOU SAID:")
    print("-" * 60)
    if transcription.strip():
        print(f'  "{transcription.strip()}"')
    else:
        print("  (No speech detected - was it silent?)")
    print("-" * 60)
    print()
    
    # Cleanup
    os.unlink(temp_path)
    
    print("=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
else:
    print("\nNo audio was recorded.")
