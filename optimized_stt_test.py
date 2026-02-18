"""
Optimized Real-time STT Test
- Uses 'base' model for better nuance
- ANSI escape codes for smooth, jitter-free visualizer
- Robust error handling to prevent 3-5s abortion
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

# Fix Windows console encoding and ANSI support
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# Settings
SAMPLE_RATE = 16000
FFT_SIZE = 1024
BAR_COUNT = 20
TEST_DURATION = 60 
TRANSCRIBE_INTERVAL = 5  # Faster updates (5s instead of 10s)

# Color constants for visualizer
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[H\033[J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

print(f"{BLUE}=" * 70)
print("OPTIMIZED STT TEST - ACCURACY & STABILITY FIX")
print("=" * 70 + RESET)

# Load a better model for "nuance"
print("\nLoading 'base' model (Higher accuracy than 'tiny')...")
try:
    model = whisper.load_model("base")
    print(f"{GREEN}[OK] Model loaded (FP32 mode){RESET}")
except Exception as e:
    print(f"{RED}[FAIL] Could not load model: {e}{RESET}")
    sys.exit(1)

# Shared state
audio_buffer = deque(maxlen=int(SAMPLE_RATE * 0.15)) 
recorded_chunks = []
recording_active = True
current_transcription = ""
transcription_lock = threading.Lock()
error_log = []

def audio_callback(indata, frames, time_info, status):
    if status:
        error_log.append(f"Stream Status: {status}")
    audio_buffer.extend(indata.copy())
    if recording_active:
        recorded_chunks.append(indata.copy())

def calculate_bars(audio_data):
    if len(audio_data) < FFT_SIZE:
        return [0] * BAR_COUNT
    window = np.hanning(FFT_SIZE)
    windowed = audio_data[-FFT_SIZE:].flatten() * window
    fft = np.abs(np.fft.rfft(windowed))
    
    bars = []
    # Logarithmic-ish grouping for better visual feel
    indices = np.linspace(0, len(fft)-1, BAR_COUNT + 1).astype(int)
    for i in range(BAR_COUNT):
        bar_val = np.mean(fft[indices[i]:indices[i+1]])
        bars.append(bar_val)
    
    max_val = max(bars) if max(bars) > 0.01 else 1.0
    return [min(1.0, v / (max_val * 0.5)) for v in bars]

def draw_visualizer(bars, volume, elapsed, last_transcription=""):
    # Move cursor to top instead of clearing screen (prevents jitter)
    sys.stdout.write("\033[H") 
    
    print(f"{BLUE}=" * 70)
    print(f"STT OPTIMIZED TEST - {TEST_DURATION - elapsed:02d}s REMAINING")
    print("=" * 70 + RESET)
    print()
    
    # Volume Meter
    vol_width = int(volume * 50)
    color = GREEN if volume < 0.4 else (YELLOW if volume < 0.7 else RED)
    vol_bar = color + "█" * vol_width + RESET + "░" * (50 - vol_width)
    print(f"INPUT LEVEL: [{vol_bar}] {int(volume*100):3d}%")
    print()
    
    # Frequency spectrum
    print("LIVE SPECTRUM:")
    bar_height = 10
    for row in range(bar_height, 0, -1):
        line = "  "
        for bar in bars:
            val = int(bar * bar_height)
            if val >= row:
                line += f"{BLUE}██ {RESET}"
            elif val > row - 0.5:
                line += f"{BLUE}▄▄ {RESET}"
            else:
                line += "   "
        print(line)
    print("  " + "▔" * (BAR_COUNT * 3))
    print()
    
    # Transcription (Word Wrapped)
    print(f"{YELLOW}TRANSCRIPTION (Live Update):{RESET}")
    print("-" * 70)
    if last_transcription:
        # Simple wrap for console
        text = last_transcription
        print(f"  > {text[:65]}")
        if len(text) > 65: print(f"    {text[65:130]}")
    else:
        print("  (Listening for speech...)")
    print("-" * 70)
    
    if error_log:
        print(f"{RED}Recent Alerts: {error_log[-1]}{RESET}")

def transcribe_worker():
    global current_transcription
    while recording_active:
        time.sleep(TRANSCRIBE_INTERVAL)
        if len(recorded_chunks) > 20: # Ensure we have enough data
            try:
                # Transcribe the most recent 15 seconds for context
                recent_data = recorded_chunks[-int(15 * SAMPLE_RATE / len(recorded_chunks[0])):]
                audio_data = np.concatenate(recent_data, axis=0).flatten().astype(np.float32)
                
                result = model.transcribe(audio_data, fp16=False, language='en')
                with transcription_lock:
                    current_transcription = result["text"].strip()
            except Exception as e:
                error_log.append(f"Worker Error: {str(e)[:40]}")

# Start Stream
sys.stdout.write(CLEAR_SCREEN + HIDE_CURSOR)
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=FFT_SIZE):
        threading.Thread(target=transcribe_worker, daemon=True).start()
        
        start_time = time.time()
        while recording_active and (time.time() - start_time) < TEST_DURATION:
            elapsed = int(time.time() - start_time)
            
            if len(audio_buffer) >= FFT_SIZE:
                audio_data = np.array(audio_buffer)
                rms = np.sqrt(np.mean(audio_data**2))
                volume = min(1.0, rms / 0.15) # Adjusted sensitivity
                bars = calculate_bars(audio_data)
                
                with transcription_lock:
                    lt = current_transcription
                
                draw_visualizer(bars, volume, elapsed, lt)
            
            time.sleep(0.05) # 20 FPS for smoothness
            
except Exception as e:
    print(f"\n{RED}CRITICAL FAILURE: {e}{RESET}")
finally:
    recording_active = False
    sys.stdout.write(SHOW_CURSOR + "\n")

print("\n" + "=" * 70)
print(f"{GREEN}FINAL ANALYSIS{RESET}")
print("=" * 70)

if recorded_chunks:
    print("Processing full session for maximum detail...")
    full_audio = np.concatenate(recorded_chunks, axis=0).flatten().astype(np.float32)
    final_result = model.transcribe(full_audio, fp16=False)
    
    print(f"\n{YELLOW}COMPLETE TRANSCRIPT:{RESET}")
    print(final_result["text"].strip() or "(Silence recorded)")
    
    # Save
    output_path = f"output/optimized_test_{int(time.time())}.wav"
    os.makedirs("output", exist_ok=True)
    audio_int16 = (full_audio * 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    print(f"\n{GREEN}Saved high-quality capture to: {output_path}{RESET}")

print("\nDone.")
