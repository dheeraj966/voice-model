"""
MASTER STT FIX - V16 MASTER-STABLE (100% Stability & Session Memory)
- Architecture: Single-Process Synchronous Engine (Bug-Fixed)
- Feature: Session-Wide Memory (Transcribes entire log at end)
- Optimization: 2.0x Gain + 0.01 Silence Threshold
- Dashboard: Perfectly Aligned Dual-Row Terminal UI
"""

import numpy as np
import sounddevice as sd
import sys
import os
import time
import wave
from collections import deque
from faster_whisper import WhisperModel

# Windows-Specific Terminal Fixes
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# Global Constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 8192       
TEST_DURATION = 300     
MODEL_SIZE = "base.en"  

# ANSI Styles
CLEAR = "\033[H\033[J"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

def draw_ui(vol, elapsed, text, status):
    """Clean, stable Terminal UI with absolute alignment."""
    sys.stdout.write("\033[H")
    print(f"{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BLUE}║{RESET} {BOLD}MASTER STT V16{RESET} | {elapsed//60:02d}m {elapsed%60:02d}s / 05:00 | Status: {status:<12} {BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    
    # Meter
    v_width = int(vol * 40)
    meter = (GREEN if vol < 0.3 else YELLOW) + "█" * v_width + RESET + "░" * (40 - v_width)
    print(f"{BLUE}║{RESET} MIC SENSITIVITY: [{meter}] {int(vol*100):3d}% {' '*2}{BLUE}║{RESET}")
    
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    print(f"{BLUE}║{RESET} {CYAN}NUANCE CAPTURE (LIVE):{RESET}{' '*46}{BLUE}║{RESET}")
    
    # Dual-Row Wrapping
    display_text = text[-130:] if len(text) > 130 else text
    line1 = display_text[:64]
    line2 = display_text[64:128]
    print(f"{BLUE}║{RESET}  > {line1:<64} {BLUE}║{RESET}")
    print(f"{BLUE}║{RESET}    {line2:<64} {BLUE}║{RESET}")
    
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    # Progress
    p = int((elapsed / TEST_DURATION) * 50)
    prog_bar = "#" * p + "·" * (50 - p)
    print(f"{BLUE}║{RESET} TOTAL SESSION  : [{prog_bar}] {int(elapsed/TEST_DURATION*100):3d}% {' '*2}{BLUE}║{RESET}")
    print(f"{BLUE}╚{'═'*68}╝{RESET}")

def main():
    print(f"{YELLOW}Waking up AI Engine (base.en)...{RESET}")
    
    # Load Model (Optimized for Intel CPU Nuance)
    try:
        # Intel Iris Xe doesn't support CUDA. Using int8 on CPU for maximum Intel efficiency.
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=8)
        print(f"{GREEN}[OK] AI Ready (Intel CPU Optimized - int8).{RESET}")
    except Exception as e:
        print(f"{RED}LOAD ERROR: {e}{RESET}")
        return

    # Initial Variables (CRITICAL: Prevents UnboundLocalError)
    full_audio_log = []
    viz_buffer = deque(maxlen=2048)
    rolling_audio = deque(maxlen=int(SAMPLE_RATE * 3.0)) 
    current_transcription = ""
    status = f"{CYAN}LISTENING{RESET}"
    last_transcription_time = time.time()
    
    def callback(indata, frames, time_info, status_msg):
        audio = indata.copy().flatten()
        viz_buffer.extend(audio)
        rolling_audio.extend(audio)
        full_audio_log.append(audio)

    print(f"{YELLOW}Opening Microphone Stream...{RESET}")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BLOCK_SIZE):
            start_time = time.time()
            sys.stdout.write(CLEAR + "\033[?25l")
            
            while (time.time() - start_time) < TEST_DURATION:
                now = time.time()
                elapsed = int(now - start_time)
                
                # TRANSCRIPTION (Every 1.2s)
                if now - last_transcription_time > 1.2:
                    status = f"{YELLOW}THINKING{RESET}"
                    audio_to_transcribe = np.array(rolling_audio, dtype=np.float32)
                    
                    # Lowered VAD Threshold for subtle nuances
                    if np.max(np.abs(audio_to_transcribe)) > 0.01:
                        # 2.0x Gain
                        audio_to_transcribe = np.clip(audio_to_transcribe * 2.0, -1.0, 1.0)
                        # Increased beam_size for nuance capturing
                        segments, _ = model.transcribe(audio_to_transcribe, beam_size=3)
                        new_text = " ".join([s.text for s in segments]).strip()
                        if new_text:
                            current_transcription = new_text
                    
                    last_transcription_time = now
                    status = f"{CYAN}LISTENING{RESET}"

                # UI RENDERING (Corrected variable name)
                if len(viz_buffer) >= 1024:
                    v_data = np.array(list(viz_buffer))
                    vol = min(1.0, np.sqrt(np.mean(v_data**2)) / 0.15)
                    draw_ui(vol, elapsed, current_transcription, status)
                
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        print(f"\n{GREEN}SESSION ENDED. Starting Deep Capture Summary...{RESET}")
        
        if full_audio_log:
            final_audio = np.concatenate(full_audio_log)
            # Save for safety
            os.makedirs("output", exist_ok=True)
            out_file = f"output/v16_master_{int(time.time())}.wav"
            with wave.open(out_file, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes((final_audio * 32767).astype(np.int16).tobytes())
            
            # FINAL FULL TRANSCRIPTION (The Latency Rule fix)
            print(f"{YELLOW}Processing 100% of recorded audio for nuance...{RESET}")
            # Transcribe the WHOLE session, not just the last snippet
            segments, _ = model.transcribe(final_audio.astype(np.float32), beam_size=3)
            final_text = " ".join([s.text for s in segments]).strip()
            
            print(f"\n{BLUE}{'═'*70}{RESET}")
            print(f"{GREEN}{BOLD}FINAL SESSION SUMMARY (Nuance Optimized):{RESET}")
            print(f"\n\"{final_text if final_text else '(No speech detected)'}\"\n")
            print(f"{BLUE}{'═'*70}{RESET}")
            print(f"Captured: {out_file}\n")
            
    sys.exit(0)

if __name__ == "__main__":
    main()
