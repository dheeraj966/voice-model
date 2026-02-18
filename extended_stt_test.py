"""
Extended Real-time STT Test with Visualizer
60-second test with live visualization and continuous transcription.
"""

import numpy as np
import sounddevice as sd
import whisper
import sys
import os
import time
import wave
from collections import deque
import threading

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Settings
SAMPLE_RATE = 16000
FFT_SIZE = 1024
BAR_COUNT = 16
TEST_DURATION = 60  # 1 minute
TRANSCRIBE_INTERVAL = 10  # Transcribe every 10 seconds

print("=" * 70)
print("EXTENDED STT TEST WITH REAL-TIME VISUALIZER")
print("=" * 70)
print("\nTest Duration: 60 seconds")
print("Transcription: Every 10 seconds")
print("\nSpeak naturally throughout the test!")
print("=" * 70)
print("\nLoading Whisper model...")

# Load Whisper model with explicit FP32 (suppresses warning)
model = whisper.load_model("tiny")
print("[OK] Model loaded (FP32 mode)")
time.sleep(1)

# Shared data
audio_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))  # 100ms for viz
recorded_chunks = []
recording_active = True
current_transcription = ""
transcription_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream."""
    if status:
        pass  # Ignore status messages
    audio_buffer.extend(indata.copy())
    if recording_active:
        recorded_chunks.append(indata.copy())

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

def draw_visualizer(bars, volume, elapsed, last_transcription=""):
    """Draw the equalizer in terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 70)
    print(f"STT TEST - {TEST_DURATION - elapsed}s remaining")
    print("=" * 70)
    print()
    
    # Volume meter
    vol_percent = int(volume * 100)
    vol_bar = "█" * int(volume * 40)
    vol_empty = "░" * (40 - len(vol_bar))
    print(f"VOLUME: [{vol_bar}{vol_empty}] {vol_percent:3d}%")
    print()
    
    # Status
    if volume > 0.1:
        status = "🎤 DETECTING SPEECH..."
    else:
        status = "⏳ Listening..."
    print(f"  {status}")
    print()
    
    # Frequency bars
    print("FREQUENCY SPECTRUM:")
    print("-" * 70)
    
    bar_height = 8
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
    
    # Last transcription
    if last_transcription:
        print("LAST TRANSCRIPTION:")
        print("-" * 70)
        print(f'  "{last_transcription}"')
        print("-" * 70)
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
    print("=" * 70)

def transcribe_audio():
    """Transcribe recorded audio in background."""
    global current_transcription, recorded_chunks
    
    while recording_active:
        time.sleep(TRANSCRIBE_INTERVAL)
        
        if len(recorded_chunks) > 0:
            # Get last 10 seconds of audio
            chunks_to_transcribe = recorded_chunks[-int(10 * SAMPLE_RATE / len(recorded_chunks[0])):]
            if len(chunks_to_transcribe) > 0:
                audio_concat = np.concatenate(chunks_to_transcribe, axis=0)
                audio_flat = audio_concat.flatten().astype(np.float32)
                
                with transcription_lock:
                    try:
                        result = model.transcribe(audio_flat, fp16=False)  # Explicit FP32
                        current_transcription = result["text"].strip()
                        print(f"\n[TRANSCRIPTION UPDATED]")
                    except Exception as e:
                        print(f"\n[Transcription error: {e}]")

print("\nStarting audio stream...")
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    blocksize=FFT_SIZE
)
stream.start()

# Start transcription thread
transcribe_thread = threading.Thread(target=transcribe_audio, daemon=True)
transcribe_thread.start()

print("Test starting in 3 seconds...")
time.sleep(3)

# Main test loop
start_time = time.time()
try:
    while recording_active and (time.time() - start_time) < TEST_DURATION:
        elapsed = int(time.time() - start_time)
        
        if len(audio_buffer) >= FFT_SIZE:
            audio_data = np.array(audio_buffer)[-FFT_SIZE:]
            rms = np.sqrt(np.mean(audio_data ** 2))
            volume = min(1.0, rms / 0.1)
            bars = calculate_bars(audio_data)
            
            with transcription_lock:
                last_transcription = current_transcription
            
            draw_visualizer(bars, volume, elapsed, last_transcription)
        
        time.sleep(0.1)
    
    recording_active = False
    
except KeyboardInterrupt:
    print("\n\nTest stopped by user.")
    recording_active = False

finally:
    stream.stop()
    stream.close()

# Final transcription
print("\n" * 2)
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print()

if len(recorded_chunks) > 0:
    audio_concat = np.concatenate(recorded_chunks, axis=0)
    audio_flat = audio_concat.flatten().astype(np.float32)
    
    print("Transcribing full recording...")
    result = model.transcribe(audio_flat, fp16=False)
    full_transcription = result["text"].strip()
    
    print()
    print("-" * 70)
    print("FULL TRANSCRIPTION:")
    print("-" * 70)
    if full_transcription:
        print(f'  "{full_transcription}"')
    else:
        print("  (No speech detected)")
    print("-" * 70)
    print()
    
    # Save audio file
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'stt_test_{int(time.time())}.wav')
    
    audio_int16 = (audio_concat * 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"Audio saved to: {output_path}")
    print(f"Recording duration: {len(audio_concat)/SAMPLE_RATE:.1f} seconds")

print()
print("=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
