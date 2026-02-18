"""
AUDIO-DECOUPLED MULTIPROCESSING STT - FINAL FIX
- Pre-loads model BEFORE audio starts (no cold start)
- Large buffer (8192 samples = 512ms window)
- Non-blocking queue (drops audio rather than blocking)
- Separate process for AI (PortAudio-safe)
- psutil priority control
"""

import multiprocessing as mp
from multiprocessing import Queue
import numpy as np
import sounddevice as sd
import sys
import os
import time
import wave
from collections import deque

# Windows optimizations
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Set process priority
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except:
        pass

# ============= CRITICAL SETTINGS =============
TEST_DURATION = 30  # Start with 30s for quick verification, change to 60 for full test
SAMPLE_RATE = 16000
BUFFER_SIZE = 8192      # 512ms window (was 64ms with 1024)
FFT_SIZE = 1024
BAR_COUNT = 10
MODEL_NAME = "tiny.en"  # English-only, 20% faster
# =============================================

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
print("AUDIO-DECOUPLED STT - FINAL FIX")
print(f"{'='*70}{RESET}")
print(f"\n{BOLD}Core Fixes:{RESET}")
print(f"  • Buffer: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLE_RATE*1000:.0f}ms window)")
print(f"  • Model: {MODEL_NAME} (English-only, optimized)")
print(f"  • Queue: maxsize=1 (drop audio, never block)")
print(f"  • Architecture: Pre-loaded model + separate AI process")
print(f"\n{CYAN}PortAudio-safe architecture!{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

# ============= PRE-LOAD MODEL (BEFORE AUDIO) =============
print(f"{YELLOW}STEP 1: Pre-loading model (BEFORE audio starts)...{RESET}")
try:
    import whisper
    print(f"  Loading {MODEL_NAME}...")
    model = whisper.load_model(MODEL_NAME)
    print(f"  {GREEN}✓ Model loaded and ready{RESET}")
except Exception as e:
    print(f"  {RED}✗ Model load failed: {e}{RESET}")
    sys.exit(1)

# ============= WORKER PROCESS =============
def ai_worker(audio_queue, result_queue, stop_event):
    """
    Separate process for AI - never blocks audio callback
    Uses non-blocking queue (drops audio if too slow)
    """
    print("[AI Worker] Started")
    
    while not stop_event.is_set():
        try:
            # Non-blocking get with timeout
            audio_data = audio_queue.get(timeout=0.5)
            
            if audio_data is None:  # Stop signal
                print("[AI Worker] Stopping...")
                break
            
            # Transcribe (takes time, but audio callback doesn't care!)
            result = model.transcribe(audio_data, fp16=False, language='en')
            text = result["text"].strip()
            
            if text:
                try:
                    result_queue.put_nowait(text)
                except:
                    pass  # Queue full, skip
        
        except Exception as e:
            # queue.Empty or timeout - continue running
            continue
    
    print("[AI Worker] Shutdown complete")

# ============= MAIN PROCESS =============
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

def draw(bars, vol, elapsed, transcription=""):
    print(CLEAR, end='')
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"STT TEST - {TEST_DURATION - elapsed}s remaining")
    print(f"Buffer: {BUFFER_SIZE} samples | Model: {MODEL_NAME}")
    print(f"{'='*70}{RESET}\n")
    
    vp = int(vol * 100)
    vb = "█" * int(vol * 40)
    ve = "░" * (40 - len(vb))
    status = "🎤" if vol > 0.1 else "👂"
    print(f"Level: [{vb}{ve}] {vp:3d}% {status}")
    print()
    
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
    
    if transcription:
        print(f"{BOLD}Latest:{RESET} \"{transcription}\"")
        print()
    
    progress = int((elapsed / TEST_DURATION) * 40)
    progress_bar = "█" * progress + "░" * (40 - progress)
    print(f"Progress: [{progress_bar}]")
    print()
    
    print(f"{BLUE}{'='*70}{RESET}")

def main():
    # Setup queues (maxsize=1 = drop audio rather than block!)
    audio_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=5)
    stop_event = mp.Event()
    
    # Start AI worker BEFORE audio
    print(f"\n{YELLOW}STEP 2: Starting AI worker process...{RESET}")
    worker = mp.Process(target=ai_worker, args=(audio_queue, result_queue, stop_event))
    worker.start()
    time.sleep(1)  # Let worker initialize
    
    # Audio state
    recording = np.zeros(int(SAMPLE_RATE * TEST_DURATION * 1.5), dtype=np.float32)
    rec_idx = 0
    viz_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))
    running = True
    latest_transcription = ""
    last_transcribe = 0
    
    # Audio callback (MUST be lightweight - no blocking!)
    def audio_callback(indata, frames, time_info, status):
        nonlocal rec_idx
        if status:
            pass  # Ignore status
        
        # Copy to recording buffer
        audio = indata.copy().flatten()
        viz_buffer.extend(audio)
        
        if rec_idx + len(audio) < len(recording):
            recording[rec_idx:rec_idx+len(audio)] = audio
            rec_idx += len(audio)
        
        # Try to queue for AI (non-blocking - drops if full!)
        try:
            audio_queue.put_nowait(audio.copy())
        except:
            pass  # Queue full - drop this chunk (audio callback must not block!)
    
    # Start audio stream (model already loaded, worker already running)
    print(f"{YELLOW}STEP 3: Starting audio stream (large buffer)...{RESET}")
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=BUFFER_SIZE,  # 512ms window!
            latency='high'  # Prioritize stability over latency
        )
        stream.start()
        print(f"{GREEN}[OK] Audio started with {BUFFER_SIZE/SAMPLE_RATE*1000:.0f}ms buffer{RESET}\n")
    except Exception as e:
        print(f"{RED}[!] Audio error: {e}{RESET}")
        stop_event.set()
        worker.terminate()
        sys.exit(1)
    
    print("Starting in 3...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    # ============= MAIN LOOP (NEVER BLOCKED) =============
    start = time.time()
    
    try:
        while (time.time() - start) < TEST_DURATION:
            elapsed = int(time.time() - start)
            
            # Get results (non-blocking)
            try:
                while True:
                    text = result_queue.get_nowait()
                    latest_transcription = text
            except:
                pass
            
            # Draw UI (always responsive)
            if len(viz_buffer) >= FFT_SIZE:
                viz = np.array(list(viz_buffer)[-FFT_SIZE:])
                rms = np.sqrt(np.mean(viz ** 2))
                vol = min(1.0, rms / 0.1)
                bars = get_bars(viz)
                draw(bars, vol, elapsed, latest_transcription)
            
            time.sleep(0.05)
        
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopped{RESET}")
    
    finally:
        running = False
        stream.stop()
        stream.close()
        
        # Stop worker
        print("\nStopping AI worker...")
        try:
            audio_queue.put_nowait(None)
            stop_event.set()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
            print(f"{GREEN}[OK] Worker stopped{RESET}")
        except:
            pass
    
    # ============= RESULTS =============
    print("\n" * 2)
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}RESULTS{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    duration = rec_idx / SAMPLE_RATE
    print(f"✓ Recorded {duration:.1f}s")
    print(f"✓ Buffer: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLE_RATE*1000:.0f}ms)")
    print(f"✓ Model: {MODEL_NAME}")
    
    print("\nFinal transcription...")
    try:
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
            print("  (No speech)")
        print(f"{'─'*70}\n")
    except Exception as e:
        print(f"{RED}[!] Error: {e}{RESET}")
    
    # Save
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'final_stt_{int(time.time())}.wav')
    
    with wave.open(outpath, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((recording[:rec_idx] * 32767).astype(np.int16).tobytes())
    
    print(f"{GREEN}✓ Saved:{RESET} {outpath}")
    print(f"{GREEN}✓ Duration:{RESET} {duration:.1f}s")
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{GREEN}COMPLETE!{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"\nRuntime: {int(time.time() - start)}s")

if __name__ == "__main__":
    mp.freeze_support()
    main()
