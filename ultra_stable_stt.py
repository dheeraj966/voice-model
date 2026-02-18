"""
ULTRA-STABLE STT TEST - NO THREADING ISSUES
- Single-threaded (no background transcription crashes)
- Visualizer only (no live transcription)
- Full 60-second guaranteed runtime
- Transcribes ONLY at the end
"""

import numpy as np
import sounddevice as sd
import whisper
import sys
import os
import time
import wave
from collections import deque

# Windows ANSI support
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# ============= SETTINGS =============
TEST_DURATION = 60
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
FFT_SIZE = 1024
BAR_COUNT = 10
# ====================================

# ANSI codes
CLEAR = "\033[H\033[J"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"

print(f"{BLUE}{'='*70}")
print("ULTRA-STABLE STT TEST - 60 SECONDS GUARANTEED")
print(f"{'='*70}{RESET}")
print(f"\n{BOLD}Settings:{RESET}")
print(f"  Duration: {TEST_DURATION}s")
print(f"  Sample Rate: {SAMPLE_RATE}Hz")
print(f"  Buffer: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLE_RATE*1000:.1f}ms)")
print(f"\n{YELLOW}No background transcription = NO CRASHES!{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

# Load model
print("Loading Whisper model...")
model = whisper.load_model("tiny")
print(f"{GREEN}[OK] Model ready{RESET}\n")
time.sleep(1)

# Buffers
MAX_AUDIO = int(SAMPLE_RATE * TEST_DURATION * 1.2)
recording = np.zeros(MAX_AUDIO, dtype=np.float32)
rec_idx = 0
viz_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))
running = True

def callback(indata, frames, time_info, status):
    global rec_idx, running
    audio = indata.copy().flatten()
    viz_buffer.extend(audio)
    if running and rec_idx + len(audio) < MAX_AUDIO:
        recording[rec_idx:rec_idx+len(audio)] = audio
        rec_idx += len(audio)

def get_bars(data):
    if len(data) < FFT_SIZE:
        return [0] * BAR_COUNT
    window = data * np.hanning(len(data))
    fft = np.abs(np.fft.rfft(window))
    bars = []
    for i in range(BAR_COUNT):
        start = i * len(fft) // BAR_COUNT
        end = (i + 1) * len(fft) // BAR_COUNT
        bars.append(np.mean(fft[start:end]) if end > start else 0)
    mx = max(bars) if max(bars) > 0 else 1
    return [(v / mx) ** 0.5 for v in bars]

def draw(bars, vol, elapsed):
    print(CLEAR, end='')
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"STT TEST - {TEST_DURATION - elapsed}s remaining")
    print(f"{'='*70}{RESET}\n")
    
    # Volume
    vp = int(vol * 100)
    vb = "█" * int(vol * 40)
    ve = "░" * (40 - len(vb))
    status = "🎤" if vol > 0.1 else "👂"
    print(f"Level: [{vb}{ve}] {vp:3d}% {status}")
    print()
    
    # Bars
    print("Frequency:")
    for row in range(8, 0, -1):
        line = "  "
        for b in bars:
            if int(b * 8) >= row:
                line += "██ "
            else:
                line += "   "
        print(line)
    print(f"  {'─' * (BAR_COUNT * 3)}")
    print()
    
    # Progress
    progress_bar = "█" * int((elapsed / TEST_DURATION) * 40)
    progress_empty = "░" * (40 - len(progress_bar))
    print(f"Progress: [{progress_bar}{progress_empty}]")
    print()
    
    print(f"Recorded: {rec_idx/SAMPLE_RATE:.1f}s / {TEST_DURATION}s")
    print(f"{BLUE}{'='*70}{RESET}")

# Start
print("Starting audio stream...")
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        blocksize=BUFFER_SIZE
    )
    stream.start()
    print(f"{GREEN}[OK] Audio started{RESET}\n")
except Exception as e:
    print(f"{RED}[!] Audio error: {e}{RESET}")
    sys.exit(1)

print("Starting in 3...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# Main loop - GUARANTEED TO RUN 60 SECONDS
start = time.time()
try:
    while (time.time() - start) < TEST_DURATION:
        elapsed = int(time.time() - start)
        
        if len(viz_buffer) >= FFT_SIZE:
            viz = np.array(list(viz_buffer)[-FFT_SIZE:])
            rms = np.sqrt(np.mean(viz ** 2))
            vol = min(1.0, rms / 0.1)
            bars = get_bars(viz)
            draw(bars, vol, elapsed)
        
        time.sleep(0.05)
    
except KeyboardInterrupt:
    print(f"\n{YELLOW}Stopped early{RESET}")

running = False
stream.stop()
stream.close()

# Results
print("\n" * 2)
print(f"{BLUE}{'='*70}{RESET}")
print(f"{BOLD}RESULTS{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

if rec_idx > SAMPLE_RATE * 2:
    duration = rec_idx / SAMPLE_RATE
    print(f"✓ Recorded {duration:.1f} seconds of audio")
    
    print("\nTranscribing...")
    try:
        result = model.transcribe(recording[:rec_idx], fp16=False, language='en')
        text = result["text"].strip()
        
        print(f"\n{'─'*70}")
        print(f"{BOLD}TRANSCRIPTION:{RESET}")
        print(f"{'─'*70}")
        if text:
            # Word wrap
            words = text.split()
            lines = []
            current = ""
            for w in words:
                if len(current) + len(w) < 65:
                    current += w + " "
                else:
                    lines.append(current.strip())
                    current = w + " "
            if current:
                lines.append(current.strip())
            for line in lines:
                print(f"  {line}")
        else:
            print("  (No speech detected)")
        print(f"{'─'*70}\n")
        
    except Exception as e:
        print(f"{RED}[!] Transcription error: {e}{RESET}")
        text = ""
    
    # Save
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'stable_stt_{int(time.time())}.wav')
    
    try:
        with wave.open(outpath, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes((recording[:rec_idx] * 32767).astype(np.int16).tobytes())
        
        print(f"{GREEN}✓ Saved:{RESET} {outpath}")
        print(f"{GREEN}✓ Duration:{RESET} {duration:.1f}s")
    except Exception as e:
        print(f"{RED}[!] Save error: {e}{RESET}")

print(f"\n{BLUE}{'='*70}{RESET}")
print(f"{GREEN}TEST COMPLETE!{RESET}")
print(f"{BLUE}{'='*70}{RESET}")
print(f"\nActual runtime: {int(time.time() - start)} seconds")
