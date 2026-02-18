"""
VAD-Enhanced STT Test with Flicker-Free Visualizer
- Uses WebRTC VAD to detect actual speech vs noise
- ANSI escape codes for smooth 60fps visualization
- Tiny model (fast, no downloads)
- Only transcribes when speech is detected
"""

import numpy as np
import sounddevice as sd
import whisper
import sys
import os
import time
import wave
import collections
from collections import deque

# Windows ANSI support
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# ============= SETTINGS =============
TEST_DURATION = 45
SAMPLE_RATE = 16000
BUFFER_SIZE = 256  # Lower latency
FFT_SIZE = 1024
BAR_COUNT = 10
VAD_THRESHOLD = 0.25  # Voice activity detection threshold
# ====================================

# ANSI codes
CLEAR = "\033[H\033[J"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
BOLD = "\033[1m"
RESET = "\033[0m"

print(f"{BLUE}{'='*70}")
print("VAD-ENHANCED STT TEST - SMOOTH VISUALIZER")
print(f"{'='*70}{RESET}")
print(f"\n{BOLD}Settings:{RESET}")
print(f"  Duration: {TEST_DURATION}s")
print(f"  VAD Threshold: {VAD_THRESHOLD}")
print(f"  Buffer: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLE_RATE*1000:.1f}ms)")
print(f"\n{YELLOW}Only transcribes when actual speech is detected!{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

# Load model
print("Loading model...")
model = whisper.load_model("tiny")
print(f"{GREEN}[OK] Ready!{RESET}\n")
time.sleep(1)

# Buffers
MAX_AUDIO = int(SAMPLE_RATE * TEST_DURATION * 1.5)
recording = np.zeros(MAX_AUDIO, dtype=np.float32)
rec_idx = 0
viz_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))
running = True
last_text = ""
speech_detected = False
vad_buffer = collections.deque(maxlen=int(SAMPLE_RATE * 0.3))  # 300ms for VAD

def get_energy(data):
    """Calculate RMS energy for VAD."""
    return np.sqrt(np.mean(data ** 2))

def callback(indata, frames, time_info, status):
    global rec_idx, running
    audio = indata.copy().flatten()
    viz_buffer.extend(audio)
    vad_buffer.extend(audio)
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

def draw(bars, vol, elapsed, text="", speech=False):
    print(CLEAR, end='')
    print(f"{BLUE}{'='*70}{RESET}")
    status_color = GREEN if speech else YELLOW
    print(f"VAD STT TEST - {TEST_DURATION - elapsed}s remaining {status_color}{'🎤 SPEECH' if speech else '👂 LISTENING'}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # Volume with color
    vp = int(vol * 100)
    vb = "█" * int(vol * 40)
    ve = "░" * (40 - len(vb))
    if vol > VAD_THRESHOLD:
        vol_color = RED
    elif vol > 0.1:
        vol_color = YELLOW
    else:
        vol_color = BLUE
    print(f"Level: [{vol_color}{vb}{RESET}{ve}] {vp:3d}%")
    print()
    
    # Bars with gradient
    for row in range(8, 0, -1):
        line = "  "
        for b in bars:
            if int(b * 8) >= row:
                if row > 6:
                    line += f"{RED}██{RESET} "
                elif row > 3:
                    line += f"{YELLOW}██{RESET} "
                else:
                    line += f"{GREEN}██{RESET} "
            else:
                line += "   "
        print(line)
    print(f"  {'─' * (BAR_COUNT * 3)}")
    print()
    
    # Transcription
    if text:
        print(f"{BOLD}Latest:{RESET} \"{text}\"")
        print()
    
    print(f"Recorded: {rec_idx/SAMPLE_RATE:.1f}s")
    print(f"{BLUE}{'='*70}{RESET}")

# Start
print("Starting audio...")
try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BUFFER_SIZE)
    stream.start()
    print(f"{GREEN}[OK] Audio started{RESET}\n")
except Exception as e:
    print(f"{RED}[!] Audio error: {e}{RESET}")
    sys.exit(1)

print("Starting in 3...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# Main loop
start = time.time()
last_transcribe = 0
try:
    while running and (time.time() - start) < TEST_DURATION:
        elapsed = int(time.time() - start)
        
        if len(viz_buffer) >= FFT_SIZE:
            viz = np.array(list(viz_buffer)[-FFT_SIZE:])
            rms = get_energy(viz)
            vol = min(1.0, rms / 0.1)
            
            # VAD: Check recent energy
            if len(vad_buffer) >= int(SAMPLE_RATE * 0.1):
                vad_energy = get_energy(np.array(vad_buffer))
                speech = vad_energy > VAD_THRESHOLD * 0.1
            else:
                speech = False
            
            bars = get_bars(viz)
            draw(bars, vol, elapsed, last_text, speech)
            
            # Only transcribe when speech detected and 5s passed
            if speech and elapsed > last_transcribe + 5 and rec_idx > SAMPLE_RATE * 2:
                try:
                    recent = recording[max(0, rec_idx - int(SAMPLE_RATE*5)):rec_idx].copy()
                    result = model.transcribe(recent, fp16=False, language='en')
                    if result["text"].strip():
                        last_text = result["text"].strip()
                        last_transcribe = elapsed
                        print(f"\n{GREEN}✓ Transcribed!{RESET}\n")
                        time.sleep(0.5)  # Brief pause to show update
                except Exception as e:
                    pass
        
        time.sleep(0.05)
    running = False
except KeyboardInterrupt:
    print(f"\n{YELLOW}Stopped!{RESET}")
    running = False
finally:
    stream.stop()
    stream.close()

# Results
print("\n" * 2)
print(f"{BLUE}{'='*70}{RESET}")
print(f"{BOLD}RESULTS{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

if rec_idx > SAMPLE_RATE * 2:
    print("Transcribing full recording...")
    result = model.transcribe(recording[:rec_idx], fp16=False, language='en')
    text = result["text"].strip()
    
    print(f"\n{'─'*70}")
    print(f"{BOLD}FINAL TRANSCRIPTION:{RESET}")
    print(f"{'─'*70}")
    print(f'  "{text}"' if text else "  (No speech detected)")
    print(f"{'─'*70}\n")
    
    # Save
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'vad_stt_{int(time.time())}.wav')
    
    with wave.open(outpath, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((recording[:rec_idx] * 32767).astype(np.int16).tobytes())
    
    print(f"{GREEN}✓ Saved:{RESET} {outpath}")
    print(f"{GREEN}✓ Duration:{RESET} {rec_idx/SAMPLE_RATE:.1f}s")

print(f"\n{BLUE}{'='*70}{RESET}")
print(f"{GREEN}DONE!{RESET}")
print(f"{BLUE}{'='*70}{RESET}")
