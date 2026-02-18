"""
Optimized Real-time STT Test - FIXED VERSION
- Proper thread safety to prevent 5s crashes
- Uses 'base' model for better nuance
- ANSI escape codes for smooth, jitter-free visualizer
- Robust error handling
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

# Windows console encoding and ANSI support
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
TRANSCRIBE_INTERVAL = 5  # Transcribe every 5 seconds

# Color constants
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[H\033[J"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

print(f"{BLUE}{'='*70}")
print("OPTIMIZED STT TEST - FIXED VERSION")
print(f"{'='*70}{RESET}")
print(f"\nDuration: {TEST_DURATION}s")
print(f"Model: base (higher accuracy)")
print(f"Transcribe interval: {TRANSCRIBE_INTERVAL}s")
print(f"\n{YELLOW}Fixed: Thread safety and error handling{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

# Load model
print("Loading 'base' model...")
try:
    model = whisper.load_model("base")
    print(f"{GREEN}[OK] Model loaded (FP32 mode){RESET}")
except Exception as e:
    print(f"{RED}[FAIL] Could not load model: {e}{RESET}")
    sys.exit(1)

# Shared state with proper synchronization
audio_buffer = deque(maxlen=int(SAMPLE_RATE * 0.15))
recorded_chunks = []
recording_active = True
current_transcription = ""
transcription_lock = threading.Lock()
error_log = []

def audio_callback(indata, frames, time_info, status):
    """Audio capture callback."""
    if status:
        pass  # Ignore status messages
    audio_buffer.extend(indata.copy())
    if recording_active:
        recorded_chunks.append(indata.copy())

def calculate_bars(audio_data):
    """Calculate frequency bars."""
    if len(audio_data) < FFT_SIZE:
        return [0] * BAR_COUNT
    window = audio_data * np.hanning(len(audio_data))
    fft = np.abs(np.fft.rfft(window))
    bars = []
    for i in range(BAR_COUNT):
        start = i * len(fft) // BAR_COUNT
        end = (i + 1) * len(fft) // BAR_COUNT
        bars.append(np.mean(fft[start:end]) if end > start else 0)
    mx = max(bars) if max(bars) > 0 else 1
    return [(v / mx) ** 0.5 for v in bars]

def draw_visualizer(bars, volume, elapsed, last_transcription=""):
    """Draw the visualizer."""
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write(f"{BLUE}{'='*70}{RESET}\n")
    sys.stdout.write(f"STT OPTIMIZED TEST - {TEST_DURATION - elapsed}s REMAINING\n")
    sys.stdout.write(f"{'='*70}{RESET}\n\n")
    
    # Volume meter
    vol_percent = int(volume * 100)
    vol_bar = "█" * int(volume * 40)
    vol_empty = "░" * (40 - len(vol_bar))
    sys.stdout.write(f"INPUT LEVEL: [{vol_bar}{vol_empty}] {vol_percent:3d}%\n\n")
    
    # Spectrum
    sys.stdout.write(f"{YELLOW}LIVE SPECTRUM:{RESET}\n")
    for row in range(10, 0, -1):
        line = "  "
        for b in bars:
            if int(b * 10) >= row:
                line += "██ "
            else:
                line += "   "
        sys.stdout.write(line + "\n")
    sys.stdout.write("  " + "▔" * (BAR_COUNT * 3) + "\n\n")
    
    # Transcription
    sys.stdout.write(f"{YELLOW}TRANSCRIPTION (Live Update):{RESET}\n")
    sys.stdout.write("-" * 70 + "\n")
    if last_transcription:
        sys.stdout.write(f"  > {last_transcription[:70]}\n")
    else:
        sys.stdout.write("  (Listening for speech...)\n")
    sys.stdout.write("-" * 70 + "\n")

def transcribe_worker():
    """Background transcription thread with proper error handling."""
    global current_transcription, recording_active
    
    while recording_active:
        time.sleep(TRANSCRIBE_INTERVAL)
        
        # Safely copy the chunks list
        chunks_copy = None
        with threading.Lock():
            if len(recorded_chunks) > 0:
                chunks_copy = list(recorded_chunks[-int(10 * SAMPLE_RATE / FFT_SIZE):])
        
        if chunks_copy and len(chunks_copy) > 0:
            try:
                # Concatenate and transcribe
                audio_data = np.concatenate(chunks_copy, axis=0).flatten().astype(np.float32)
                
                # Ensure we have valid audio
                if len(audio_data) > SAMPLE_RATE:  # At least 1 second
                    result = model.transcribe(audio_data, fp16=False, language='en')
                    
                    with transcription_lock:
                        current_transcription = result["text"].strip()
                    
                    sys.stdout.write(f"\n{GREEN}✓ Transcription updated{RESET}\n")
                    
            except Exception as e:
                error_msg = f"Worker Error: {str(e)[:50]}"
                error_log.append(error_msg)
                sys.stdout.write(f"\n{RED}{error_msg}{RESET}\n")

# Start Stream
sys.stdout.write(CLEAR_SCREEN + HIDE_CURSOR)
stream = None
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=FFT_SIZE
    )
    stream.start()
    
    # Start transcription thread
    transcribe_thread = threading.Thread(target=transcribe_worker, daemon=True)
    transcribe_thread.start()
    
    sys.stdout.write(f"{GREEN}[OK] Audio stream started{RESET}\n")
    time.sleep(1)
    
    start_time = time.time()
    
    while recording_active and (time.time() - start_time) < TEST_DURATION:
        elapsed = int(time.time() - start_time)
        
        if len(audio_buffer) >= FFT_SIZE:
            audio_data = np.array(audio_buffer)
            rms = np.sqrt(np.mean(audio_data**2))
            volume = min(1.0, rms / 0.15)
            bars = calculate_bars(audio_data)
            
            with transcription_lock:
                lt = current_transcription
            
            draw_visualizer(bars, volume, elapsed, lt)
        
        time.sleep(0.05)  # 20 FPS
    
    recording_active = False
    
except KeyboardInterrupt:
    sys.stdout.write(f"\n{YELLOW}Stopped by user{RESET}\n")
    recording_active = False
except Exception as e:
    sys.stdout.write(f"\n{RED}CRITICAL FAILURE: {e}{RESET}\n")
    import traceback
    traceback.print_exc()
finally:
    recording_active = False
    if stream:
        stream.stop()
        stream.close()
    sys.stdout.write(SHOW_CURSOR + "\n")

# Results
print("\n" + "=" * 70)
print(f"{GREEN}FINAL ANALYSIS{RESET}")
print("=" * 70)

if len(recorded_chunks) > 0:
    print("Processing full session...")
    try:
        full_audio = np.concatenate(recorded_chunks, axis=0).flatten().astype(np.float32)
        print(f"Audio length: {len(full_audio)/SAMPLE_RATE:.1f}s")
        print("Transcribing...")
        
        final_result = model.transcribe(full_audio, fp16=False, language='en')
        
        print(f"\n{YELLOW}COMPLETE TRANSCRIPT:{RESET}")
        print(final_result["text"].strip() or "(Silence recorded)")
        
        # Save
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/optimized_stt_{int(time.time())}.wav"
        
        audio_int16 = (full_audio * 32767).astype(np.int16)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"\n{GREEN}✓ Saved: {output_path}{RESET}")
        
    except Exception as e:
        print(f"{RED}Error processing final audio: {e}{RESET}")
        if error_log:
            print(f"\n{YELLOW}Error log:{RESET}")
            for err in error_log:
                print(f"  - {err}")

print("\n" + "=" * 70)
print(f"{GREEN}TEST COMPLETE!{RESET}")
print("=" * 70)
