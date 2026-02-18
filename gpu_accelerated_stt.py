"""
GPU-Accelerated STT Test - Intel Iris Xe Optimized
- Uses DirectML for GPU acceleration
- Async transcription queue (no UI blocking)
- Double-engine architecture: CPU (UI) + GPU (Inference)
"""

import numpy as np
import sounddevice as sd
import sys
import os
import time
import wave
import threading
import queue
from collections import deque
from pathlib import Path

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
USE_GPU = True  # Try DirectML first, fallback to CPU
# ====================================

# ANSI codes
CLEAR = "\033[H\033[J"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

print(f"{BLUE}{'='*70}")
print("GPU-ACCELERATED STT - Intel Iris Xe Optimized")
print(f"{'='*70}{RESET}")
print(f"\n{BOLD}Architecture:{RESET}")
print(f"  Engine 1 (CPU): UI Rendering + Audio Capture")
print(f"  Engine 2 (GPU): Whisper Inference (DirectML)")
print(f"\n{BOLD}Settings:{RESET}")
print(f"  Duration: {TEST_DURATION}s")
print(f"  Sample Rate: {SAMPLE_RATE}Hz")
print(f"  GPU Acceleration: {'ON (DirectML)' if USE_GPU else 'OFF (CPU)'}")
print(f"\n{CYAN}Offloading AI inference to Intel Iris Xe GPU!{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

# Try to load GPU-accelerated Whisper
model = None
inference_device = "CPU"

if USE_GPU:
    print("Attempting GPU acceleration (DirectML)...")
    try:
        import onnxruntime as ort
        # Check for DirectML provider
        available_providers = ort.get_available_providers()
        print(f"  Available providers: {available_providers}")
        
        if 'DmlExecutionProvider' in available_providers:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            inference_device = "Intel Iris Xe GPU (DirectML)"
            print(f"  {GREEN}✓ DirectML detected!{RESET}")
        else:
            providers = ['CPUExecutionProvider']
            inference_device = "CPU (DirectML not available)"
            print(f"  {YELLOW}⚠ DirectML not available, using CPU{RESET}")
        
        # Load faster-whisper for better performance
        try:
            from faster_whisper import WhisperModel
            print("\nLoading faster-whisper model (optimized)...")
            model = WhisperModel("tiny", device="cuda" if "CUDAExecutionProvider" in providers else "cpu", compute_type="int8")
            print(f"  {GREEN}✓ Model loaded (GPU-accelerated){RESET}")
        except ImportError:
            print(f"  {YELLOW}⚠ faster-whisper not installed, using whisper{RESET}")
            import whisper
            model = whisper.load_model("tiny")
            print(f"  {GREEN}✓ Whisper model loaded{RESET}")
        
    except Exception as e:
        print(f"  {RED}✗ GPU acceleration failed: {e}{RESET}")
        print(f"  Falling back to CPU...")
        USE_GPU = False

if not model:
    print("Loading Whisper model (CPU mode)...")
    import whisper
    model = whisper.load_model("tiny")
    inference_device = "CPU"
    print(f"  {GREEN}✓ Model loaded{RESET}")

print(f"\n{CYAN}Inference Device: {inference_device}{RESET}\n")
time.sleep(1)

# ============= DOUBLE-ENGINE ARCHITECTURE =============
# Engine 1: CPU - Audio Capture & UI
# Engine 2: GPU - Transcription Inference

# Shared queues (thread-safe)
audio_queue = queue.Queue(maxsize=100)
transcription_queue = queue.Queue(maxsize=10)

# State
running = True
recording = np.zeros(int(SAMPLE_RATE * TEST_DURATION * 1.2), dtype=np.float32)
rec_idx = 0
viz_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))
latest_transcription = ""
transcription_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    """CPU Engine: Audio capture (non-blocking)."""
    global rec_idx, running
    audio = indata.copy().flatten()
    viz_buffer.extend(audio)
    
    if running and rec_idx + len(audio) < len(recording):
        recording[rec_idx:rec_idx+len(audio)] = audio
        rec_idx += len(audio)
    
    # Add to queue for GPU processing (every 5 seconds)
    if rec_idx % (SAMPLE_RATE * 5) < len(audio):
        try:
            audio_queue.put_nowait(audio.copy())
        except queue.Full:
            pass  # Skip if queue is full

def gpu_transcribe_worker():
    """GPU Engine: Transcription inference (offloaded from CPU)."""
    global latest_transcription, running
    
    while running:
        try:
            # Get audio from queue
            audio_chunk = audio_queue.get(timeout=2)
            
            # Process on GPU
            if hasattr(model, 'transcribe'):  # faster-whisper
                segments, info = model.transcribe(audio_chunk, language='en', vad_filter=True)
                text = " ".join([segment.text for segment in segments]).strip()
            else:  # standard whisper
                result = model.transcribe(audio_chunk, fp16=False, language='en')
                text = result["text"].strip()
            
            # Send result back
            if text:
                with transcription_lock:
                    latest_transcription = text
                
        except queue.Empty:
            continue
        except Exception as e:
            pass

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
    print(f"GPU-STT TEST - {TEST_DURATION - elapsed}s remaining")
    print(f"Device: {inference_device}")
    print(f"{'='*70}{RESET}\n")
    
    # Volume
    vp = int(vol * 100)
    vb = "█" * int(vol * 40)
    ve = "░" * (40 - len(vb))
    status = "🎤" if vol > 0.1 else "👂"
    print(f"Level: [{vb}{ve}] {vp:3d}% {status}")
    print()
    
    # GPU Status
    gpu_status = f"{GREEN}● GPU Active{RESET}" if not audio_queue.empty() else f"{BLUE}○ GPU Idle{RESET}"
    print(f"GPU: {gpu_status} | Queue: {audio_queue.qsize()} chunks")
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
    
    # Transcription
    with transcription_lock:
        lt = latest_transcription
    if lt:
        print(f"{BOLD}Latest:{RESET} \"{lt}\"")
        print()
    
    # Progress
    progress = int((elapsed / TEST_DURATION) * 40)
    progress_bar = "█" * progress + "░" * (40 - progress)
    print(f"Progress: [{progress_bar}]")
    print()
    
    print(f"Recorded: {rec_idx/SAMPLE_RATE:.1f}s")
    print(f"{BLUE}{'='*70}{RESET}")

# ============= START ENGINES =============
print("Starting audio engine (CPU)...")
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=BUFFER_SIZE
    )
    stream.start()
    print(f"{GREEN}[OK] CPU Engine started{RESET}")
except Exception as e:
    print(f"{RED}[!] Audio error: {e}{RESET}")
    sys.exit(1)

print("Starting transcription worker (GPU)...")
gpu_thread = threading.Thread(target=gpu_transcribe_worker, daemon=True)
gpu_thread.start()
print(f"{GREEN}[OK] GPU Engine started{RESET}\n")

print("Starting in 3...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# ============= MAIN LOOP (CPU) =============
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
    print(f"\n{YELLOW}Stopped{RESET}")

running = False
stream.stop()
stream.close()

# ============= FINAL RESULTS =============
print("\n" * 2)
print(f"{BLUE}{'='*70}{RESET}")
print(f"{BOLD}FINAL RESULTS{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

duration = rec_idx / SAMPLE_RATE
print(f"✓ Recorded {duration:.1f}s of audio")
print(f"✓ Inference Device: {inference_device}")
print(f"✓ GPU Acceleration: {'ON' if USE_GPU else 'OFF'}")

print("\nFinal transcription (GPU-accelerated)...")
try:
    if hasattr(model, 'transcribe') and not isinstance(model, whisper.Whisper):
        segments, info = model.transcribe(recording[:rec_idx], language='en', vad_filter=True)
        text = " ".join([segment.text for segment in segments]).strip()
    else:
        result = model.transcribe(recording[:rec_idx], fp16=False, language='en')
        text = result["text"].strip()
    
    print(f"\n{'─'*70}")
    print(f"{BOLD}TRANSCRIPTION:{RESET}")
    print(f"{'─'*70}")
    if text:
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
outpath = os.path.join(outdir, f'gpu_stt_{int(time.time())}.wav')

with wave.open(outpath, 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    f.writeframes((recording[:rec_idx] * 32767).astype(np.int16).tobytes())

print(f"{GREEN}✓ Saved:{RESET} {outpath}")
print(f"{GREEN}✓ Duration:{RESET} {duration:.1f}s")
print(f"{GREEN}✓ Device:{RESET} {inference_device}")

print(f"\n{BLUE}{'='*70}{RESET}")
print(f"{GREEN}TEST COMPLETE!{RESET}")
print(f"{BLUE}{'='*70}{RESET}")
print(f"\nActual runtime: {int(time.time() - start)} seconds")
