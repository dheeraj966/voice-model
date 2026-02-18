"""
MULTIPROCESSING STT TEST - Zero Jitter, No Abortion
- Separate Process for AI inference (bypasses Python GIL)
- Main Process: UI + Audio (100% responsive)
- Worker Process: Whisper inference (isolated, never blocks UI)
- Queue-based IPC (Inter-Process Communication)
"""

import multiprocessing as mp
import numpy as np
import sounddevice as sd
import sys
import os
import time
import wave
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

# ============= WORKER PROCESS =============
def inference_worker(audio_queue, result_queue, ready_queue):
    """
    Runs in SEPARATE PROCESS - isolated from UI/Audio
    Heavy AI load never blocks the main process
    """
    try:
        # Import inside worker (isolated memory space)
        import whisper
        
        # Load model (main process keeps running!)
        print("[Worker] Loading Whisper model...")
        model = whisper.load_model("tiny")
        print("[Worker] Model loaded - ready for inference")
        
        # Signal ready
        ready_queue.put("READY")
        
        while True:
            # Non-blocking queue check
            try:
                audio_data = audio_queue.get_nowait()
                
                # Check for sentinel (stop signal)
                if audio_data is None:
                    print("[Worker] Received stop signal")
                    break
                
                # Perform inference (takes time, but main process doesn't care!)
                result = model.transcribe(audio_data, fp16=False, language='en')
                text = result["text"].strip()
                
                if text:
                    try:
                        result_queue.put_nowait(text)
                    except:
                        pass  # Queue full, skip
                
            except Exception as e:
                # queue.Empty or any other error
                time.sleep(0.1)  # Prevent busy waiting
                continue
        
        print("[Worker] Shutting down...")
        
    except Exception as e:
        print(f"[Worker] Fatal error: {e}")
        try:
            ready_queue.put(f"ERROR: {e}")
        except:
            pass

# ============= MAIN PROCESS =============
def get_bars(data):
    """Calculate frequency bars for visualizer."""
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

def draw(bars, vol, elapsed, transcription="", worker_status="Unknown"):
    """Draw UI (100% responsive, never blocked by AI)."""
    print(CLEAR, end='')
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"MULTIPROCESSING STT - {TEST_DURATION - elapsed}s remaining")
    print(f"Worker: {worker_status}")
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
    
    # Transcription
    if transcription:
        print(f"{BOLD}Latest:{RESET} \"{transcription}\"")
        print()
    
    # Progress
    progress = int((elapsed / TEST_DURATION) * 40)
    progress_bar = "█" * progress + "░" * (40 - progress)
    print(f"Progress: [{progress_bar}]")
    print()
    
    print(f"{BLUE}{'='*70}{RESET}")

def main():
    print(f"{BLUE}{'='*70}")
    print("MULTIPROCESSING STT - Zero Jitter Architecture")
    print(f"{'='*70}{RESET}")
    print(f"\n{BOLD}Architecture:{RESET}")
    print(f"  • Main Process: UI Rendering + Audio Capture")
    print(f"  • Worker Process: Whisper Inference (isolated)")
    print(f"  • IPC: Multiprocessing Queues (GIL-bypassing)")
    print(f"\n{CYAN}Separate processes = Zero blocking!{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # Setup Multiprocessing
    print("Setting up multiprocessing...")
    audio_queue = mp.Queue(maxsize=50)
    result_queue = mp.Queue(maxsize=10)
    ready_queue = mp.Queue()
    
    # Start AI Worker Process
    print("Starting worker process...")
    worker_process = mp.Process(target=inference_worker, args=(audio_queue, result_queue, ready_queue))
    worker_process.start()
    print(f"{GREEN}[OK] Worker process started (PID: {worker_process.pid}){RESET}")
    
    # Wait for worker to be ready
    print("Waiting for worker to load model...")
    try:
        status = ready_queue.get(timeout=30)
        if status.startswith("ERROR"):
            print(f"{RED}[!] Worker error: {status}{RESET}")
        else:
            print(f"{GREEN}[OK] Worker ready: {status}{RESET}\n")
    except:
        print(f"{YELLOW}[!] Worker timeout - continuing anyway{RESET}\n")
    
    # Audio state (main process)
    recording = np.zeros(int(SAMPLE_RATE * TEST_DURATION * 1.2), dtype=np.float32)
    rec_idx = 0
    viz_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))
    running = True
    latest_transcription = ""
    worker_status = "Loading..."
    
    # Audio callback (MUST be lightweight)
    def audio_callback(indata, frames, time_info, status):
        nonlocal rec_idx, running
        audio = indata.copy().flatten()
        viz_buffer.extend(audio)
        
        if running and rec_idx + len(audio) < len(recording):
            recording[rec_idx:rec_idx+len(audio)] = audio
            rec_idx += len(audio)
    
    # Start audio stream
    print("Starting audio stream...")
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=BUFFER_SIZE
        )
        stream.start()
        print(f"{GREEN}[OK] Audio stream started{RESET}\n")
    except Exception as e:
        print(f"{RED}[!] Audio error: {e}{RESET}")
        worker_process.terminate()
        sys.exit(1)
    
    print("Starting in 3...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    # ============= MAIN LOOP (100% RESPONSIVE) =============
    start = time.time()
    last_transcribe = 0
    
    try:
        while (time.time() - start) < TEST_DURATION:
            elapsed = int(time.time() - start)
            
            # Check worker status
            worker_status = "Running" if worker_process.is_alive() else "Stopped"
            
            # Get transcription results (non-blocking)
            try:
                while True:
                    text = result_queue.get_nowait()
                    latest_transcription = text
            except:
                pass
            
            # Send audio to worker (every 5 seconds)
            if elapsed > 0 and elapsed % 5 == 0 and elapsed != last_transcribe:
                last_transcribe = elapsed
                if rec_idx > SAMPLE_RATE * 2:
                    # Get recent audio
                    recent_start = max(0, rec_idx - int(SAMPLE_RATE * 5))
                    recent_audio = recording[recent_start:rec_idx].copy()
                    
                    # Send to worker (non-blocking)
                    try:
                        audio_queue.put_nowait(recent_audio)
                        worker_status = "Processing..."
                    except:
                        worker_status = "Queue full"
            
            # Draw UI (always responsive!)
            if len(viz_buffer) >= FFT_SIZE:
                viz = np.array(list(viz_buffer)[-FFT_SIZE:])
                rms = np.sqrt(np.mean(viz ** 2))
                vol = min(1.0, rms / 0.1)
                bars = get_bars(viz)
                draw(bars, vol, elapsed, latest_transcription, worker_status)
            
            time.sleep(0.05)  # 20 FPS
        
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopped by user{RESET}")
    
    finally:
        running = False
        stream.stop()
        stream.close()
        
        # Stop worker
        print("\nStopping worker process...")
        try:
            audio_queue.put_nowait(None)  # Sentinel
            worker_process.join(timeout=5)
            if worker_process.is_alive():
                worker_process.terminate()
            print(f"{GREEN}[OK] Worker stopped{RESET}")
        except:
            pass
    
    # ============= FINAL RESULTS =============
    print("\n" * 2)
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}FINAL RESULTS{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    duration = rec_idx / SAMPLE_RATE
    print(f"✓ Recorded {duration:.1f}s of audio")
    print(f"✓ Architecture: Multiprocessing (GIL-bypassed)")
    print(f"✓ Worker PID: {worker_process.pid}")
    
    print("\nFinal transcription...")
    try:
        # One final transcription in main process
        import whisper
        model = whisper.load_model("tiny")
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
    
    # Save
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'multiprocessing_stt_{int(time.time())}.wav')
    
    with wave.open(outpath, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((recording[:rec_idx] * 32767).astype(np.int16).tobytes())
    
    print(f"{GREEN}✓ Saved:{RESET} {outpath}")
    print(f"{GREEN}✓ Duration:{RESET} {duration:.1f}s")
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{GREEN}TEST COMPLETE!{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"\nActual runtime: {int(time.time() - start)} seconds")

if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()
