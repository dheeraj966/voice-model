"""
MASTER STT FIX - V9 CLEAN-EXIT (Production Release)
- Architecture: Queue De-registration (Prevents terminal hang)
- Shutdown: Aggressive Kill + Join (Zero lingering processes)
- Feedback: Deep Analysis Progress (Transparency)
- UI: Monospaced Dashboard + Fast Refresh
"""

import multiprocessing as mp
import numpy as np
import sounddevice as sd
import sys
import os
import time
import wave
from collections import deque

# Windows-Specific Environment Fixes
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# Global Constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096
TEST_DURATION = 60
MODEL_SIZE = "tiny.en" 
FINAL_MODEL = "base.en"

# ANSI Styles
CLEAR = "\033[H\033[J"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

# =============================================================================
# WORKER PROCESS: CLEAN-LINK
# =============================================================================
def clean_link_worker(audio_queue, result_pipe, stop_event):
    """AI engine with clean-exit support."""
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=4)
    except:
        import whisper
        model = whisper.load_model(MODEL_SIZE)

    result_pipe.send("READY")

    while not stop_event.is_set():
        try:
            if not audio_queue.empty():
                audio_chunk = audio_queue.get()
                if hasattr(model, 'transcribe'):
                    segments, _ = model.transcribe(audio_chunk, beam_size=1)
                    text = " ".join([s.text for s in segments]).strip()
                else:
                    result = model.transcribe(audio_chunk, fp16=False)
                    text = result["text"].strip()
                result_pipe.send(text if text else "[NO_SPEECH]")
            else:
                time.sleep(0.05)
        except EOFError: break
        except Exception: continue

# =============================================================================
# UI ENGINE
# =============================================================================
def draw_ui(vol, elapsed, transcription, status):
    sys.stdout.write("\033[H")
    print(f"{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BLUE}║{RESET} {BOLD}VOICE AGENT STT PRO{RESET} | {TEST_DURATION - elapsed:02d}s REMAINING | Status: {status:<10} {BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    
    # Meter
    v_width = int(vol * 40)
    meter = (GREEN if vol < 0.3 else YELLOW) + "█" * v_width + RESET + "░" * (40 - v_width)
    print(f"{BLUE}║{RESET} MIC LEVEL    : [{meter}] {int(vol*100):3d}% {BLUE}║{RESET}")
    
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    print(f"{BLUE}║{RESET} {CYAN}LIVE TRANSCRIPTION (REAL-TIME):{RESET}{' '*37}{BLUE}║{RESET}")
    
    # Wrapped text
    text = transcription if transcription else "Listening..."
    wrapped = text[-130:] if len(text) > 130 else text
    line1 = wrapped[:64]
    line2 = wrapped[64:128]
    print(f"{BLUE}║{RESET}  > {line1:<64} {BLUE}║{RESET}")
    print(f"{BLUE}║{RESET}    {line2:<64} {BLUE}║{RESET}")
    
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    p = int((elapsed / TEST_DURATION) * 50)
    print(f"{BLUE}║{RESET} TOTAL PROG   : [{'#'*p}{'·'*(50-p)}] {int(elapsed/TEST_DURATION*100):3d}% {' '*2}{BLUE}║{RESET}")
    print(f"{BLUE}╚{'═'*68}╝{RESET}")

def main():
    audio_queue = mp.Queue(maxsize=1)
    parent_conn, child_conn = mp.Pipe()
    stop_event = mp.Event()
    
    worker = mp.Process(target=clean_link_worker, args=(audio_queue, child_conn, stop_event))
    worker.daemon = True
    worker.start()
    
    print(f"{YELLOW}Initializing High-Priority Engine...{RESET}")
    while True:
        if parent_conn.poll(0.1):
            if parent_conn.recv() == "READY": break

    full_audio_log = []
    viz_deque = deque(maxlen=2048)
    rolling_audio = deque(maxlen=int(SAMPLE_RATE * 2.5))
    current_text = ""
    status_str = f"{CYAN}IDLE{RESET}"
    last_push = time.time()
    
    def callback(indata, frames, time_info, status_msg):
        audio = indata.copy().flatten()
        viz_deque.extend(audio)
        rolling_audio.extend(audio)
        full_audio_log.append(audio)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BLOCK_SIZE):
            start_time = time.time()
            sys.stdout.write(CLEAR + "\033[?25l")
            
            while (time.time() - start_time) < TEST_DURATION:
                now = time.time()
                elapsed = int(now - start_time)
                
                # Push every 0.6s
                if now - last_push > 0.6:
                    if len(rolling_audio) >= SAMPLE_RATE:
                        try:
                            audio_queue.put_nowait(np.array(rolling_audio, dtype=np.float32))
                            status_str = f"{YELLOW}ANALYZING{RESET}"
                            last_push = now
                        except: pass
                
                # Get Result
                if parent_conn.poll(0):
                    msg = parent_conn.recv()
                    if msg == "[NO_SPEECH]": status_str = f"{CYAN}LISTENING{RESET}"
                    else:
                        current_text = msg
                        status_str = f"{GREEN}CAPTURED{RESET}"
                
                # UI
                if len(viz_deque) >= 1024:
                    v_data = np.array(viz_deque)
                    vol = min(1.0, np.sqrt(np.mean(v_data**2)) / 0.15)
                    draw_ui(vol, elapsed, current_text, status_str)
                
                time.sleep(0.05)
                if status_str == f"{GREEN}CAPTURED{RESET}" and now - last_push > 0.4:
                    status_str = f"{CYAN}LISTENING{RESET}"

    except KeyboardInterrupt: pass
    finally:
        # --- PHASE 1: HARDWARE CLEANUP ---
        sys.stdout.write("\033[?25h\n")
        stop_event.set()
        
        # CRITICAL: Prevent Windows from hanging on the Queue thread
        audio_queue.cancel_join_thread()
        worker.kill()
        worker.join(timeout=1)
        
        # --- PHASE 2: FINAL DEEP ANALYSIS ---
        if full_audio_log:
            print(f"\n{GREEN}TEST ENDED. Starting Deep Analysis (Model: {FINAL_MODEL})...{RESET}")
            final_audio = np.concatenate(full_audio_log)
            
            # Save Reference
            os.makedirs("output", exist_ok=True)
            out_file = f"output/final_capture_{int(time.time())}.wav"
            with wave.open(out_file, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes((final_audio * 32767).astype(np.int16).tobytes())
            
            import whisper
            print(f"{YELLOW}Loading Final Model (Accuracy Pass)...{RESET}")
            model = whisper.load_model(FINAL_MODEL)
            print(f"{YELLOW}Transcribing 60s recording (Please wait)...{RESET}")
            result = model.transcribe(final_audio.astype(np.float32), fp16=False)
            
            # Final Result Display
            print("\n" + "═"*70)
            print(f"{GREEN}{BOLD}FINAL DEEP TRANSCRIPTION RESULT:{RESET}")
            print("═"*70)
            print(f"\n\"{result['text'].strip()}\"\n")
            print("═"*70)
            print(f"File: {out_file}\n")
        
        # --- PHASE 3: FORCE EXIT ---
        print(f"{GREEN}All processes terminated. Control returned to terminal.{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    mp.freeze_support()
    main()
