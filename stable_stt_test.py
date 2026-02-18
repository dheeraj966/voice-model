"""
Stable STT Test with Visualizer - Optimized for Windows
- Fixed buffer issues preventing crashes
- Uses tiny model (no downloads during test)
- Configurable duration (default 30 seconds)
- Smooth real-time visualization
"""

import numpy as np
import sounddevice as sd
import whisper
import sys
import os
import time
import wave
from collections import deque

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============= SETTINGS =============
TEST_DURATION = 30  # seconds (reduce if test crashes)
SAMPLE_RATE = 16000
BUFFER_SIZE = 512   # Lower = smoother, higher = more stable
FFT_SIZE = 1024
BAR_COUNT = 12
# ====================================

print("=" * 70)
print("STABLE STT TEST WITH VISUALIZER")
print("=" * 70)
print(f"\nSettings:")
print(f"  Duration: {TEST_DURATION}s")
print(f"  Sample Rate: {SAMPLE_RATE}Hz")
print(f"  Buffer: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLE_RATE*1000:.1f}ms)")
print(f"\nSpeak when you see '🎤 SPEECH DETECTED'")
print("=" * 70)

# Load tiny model (fast, no surprises)
print("\nLoading model...")
model = whisper.load_model("tiny")
print("[OK] Ready!")
time.sleep(1)

# Buffers
MAX_AUDIO = int(SAMPLE_RATE * TEST_DURATION * 1.5)
recording = np.zeros(MAX_AUDIO, dtype=np.float32)
rec_idx = 0
viz_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))
running = True
last_text = ""

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

def draw(bars, vol, elapsed, text=""):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 70)
    print(f"STT TEST - {TEST_DURATION - elapsed}s remaining")
    print("=" * 70)
    print()
    
    # Volume
    vp = int(vol * 100)
    vb = "█" * int(vol * 40)
    ve = "░" * (40 - len(vb))
    status = "🎤 SPEECH!" if vol > 0.15 else "⏳ Listening..."
    print(f"Volume: [{vb}{ve}] {vp:3d}% {status}")
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
    print("  " + "─" * (BAR_COUNT * 3))
    print()
    
    # Text
    if text:
        print(f'Latest: "{text}"')
        print()
    
    print(f"Recorded: {rec_idx/SAMPLE_RATE:.1f}s")
    print("=" * 70)

# Start stream
print("Starting audio...")
try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BUFFER_SIZE)
    stream.start()
    print("[OK] Audio started")
except Exception as e:
    print(f"[!] Audio error: {e}")
    print("Check microphone permissions!")
    sys.exit(1)

print("\nStarting in 3...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# Main loop
start = time.time()
try:
    while running and (time.time() - start) < TEST_DURATION:
        elapsed = int(time.time() - start)
        
        if len(viz_buffer) >= FFT_SIZE:
            viz = np.array(list(viz_buffer)[-FFT_SIZE:])
            rms = np.sqrt(np.mean(viz ** 2))
            vol = min(1.0, rms / 0.05)
            bars = get_bars(viz)
            draw(bars, vol, elapsed, last_text)
        
        # Transcribe every 5 seconds
        if elapsed > 0 and elapsed % 5 == 0 and rec_idx > SAMPLE_RATE * 2:
            try:
                recent = recording[max(0, rec_idx - int(SAMPLE_RATE*5)):rec_idx].copy()
                result = model.transcribe(recent, fp16=False, language='en')
                if result["text"].strip():
                    last_text = result["text"].strip()
            except:
                pass
        
        time.sleep(0.05)
    running = False
except KeyboardInterrupt:
    print("\nStopped!")
    running = False
finally:
    stream.stop()
    stream.close()

# Results
print("\n" * 2)
print("=" * 70)
print("RESULTS")
print("=" * 70)

if rec_idx > SAMPLE_RATE * 2:
    print("\nTranscribing full recording...")
    result = model.transcribe(recording[:rec_idx], fp16=False, language='en')
    text = result["text"].strip()
    
    print("\n" + "-" * 70)
    print("TRANSCRIPTION:")
    print("-" * 70)
    print(f'  "{text}"' if text else "  (No speech detected)")
    print("-" * 70)
    
    # Save
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'stt_{int(time.time())}.wav')
    
    with wave.open(outpath, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((recording[:rec_idx] * 32767).astype(np.int16).tobytes())
    
    print(f"\nSaved: {outpath}")
    print(f"Duration: {rec_idx/SAMPLE_RATE:.1f}s")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
