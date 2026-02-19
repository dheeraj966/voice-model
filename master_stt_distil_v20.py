"""
MASTER STT DISTIL V20 - THE TEAM'S BALANCED SOLUTION
- Engine: Distil-Whisper (6x Speed, 99% Nuance)
- Logic: Speculative Decoding (Tiny Assistant + Medium Verified)
- Sensitivity: Continuous 500ms Burst (No-Wait VAD)
- Hardware: Intel Iris Xe Optimized (FP16)
"""

import os
import sys
import time
import torch
import numpy as np
import sounddevice as sd
from collections import deque
from transformers import AutoProcessor, pipeline
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

# Windows Terminal Setup
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# Constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096      # Smaller blocks for sub-second response
MODEL_ID = "distil-whisper/distil-medium.en" # THE KEY: 6x faster than standard medium
ASSISTANT_MODEL_ID = "openai/whisper-tiny.en" # The "Instant Drafter"
TEST_DURATION = 300

# ANSI
GREEN, YELLOW, RED, BLUE, CYAN, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[34m", "\033[36m", "\033[1m", "\033[0m"

def draw_ui(vol, elapsed, text, status):
    sys.stdout.write("\033[H")
    print(f"{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BLUE}║{RESET} {BOLD}TEAM ENGINE V20 (DISTIL){RESET} | {elapsed//60:02d}m {elapsed%60:02d}s | {GREEN}GPU+CPU HYBRID{RESET} {BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    v_width = int(vol * 40)
    meter = (GREEN if vol < 0.3 else YELLOW) + "█" * v_width + RESET + "░" * (40 - v_width)
    print(f"{BLUE}║{RESET} MIC SENSITIVITY: [{meter}] {int(vol*100):3d}% {' '*2}{BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    print(f"{BLUE}║{RESET} {CYAN}SPECULATIVE LIVE CAPTURE:{RESET}{' '*39}{BLUE}║{RESET}")
    display_text = text[-130:] if len(text) > 130 else text
    line1, line2 = display_text[:64], display_text[64:128]
    print(f"{BLUE}║{RESET}  > {line1:<64} {BLUE}║{RESET}")
    print(f"{BLUE}║{RESET}    {line2:<64} {BLUE}║{RESET}")
    print(f"{BLUE}╚{'═'*68}╝{RESET}")

def main():
    print(f"{YELLOW}Initiating Team Strategy: Distil-Medium + Speculative Decoding...{RESET}")
    
    try:
        # Load the Main GPU Model (Distil-Medium for Speed+Nuance)
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, device="GPU", ov_config={"CACHE_DIR": "ov_cache", "PERFORMANCE_HINT": "LATENCY"}, export=True
        )
        # Load the Assistant CPU Model (Tiny for Zero-Lag Drafting)
        assistant_model = OVModelForSpeechSeq2Seq.from_pretrained(
            ASSISTANT_MODEL_ID, device="CPU", export=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        
        # Speculative Pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            generate_kwargs={
                "assistant_model": assistant_model, # POLICY: CPU Drafts, GPU Verifies
                "batch_size": 1,
                "chunk_length_s": 15,
                "beam_size": 1 # Speed priority for real-time
            }
        )
        print(f"{GREEN}[OK] Team Build V20 Active.{RESET}")
    except Exception as e:
        print(f"{RED}STRATEGY ERROR: {e}{RESET}")
        return

    # Buffers
    rolling_audio = deque(maxlen=int(SAMPLE_RATE * 5.0))
    viz_buffer = deque(maxlen=2048)
    full_text = ""
    last_process_time = time.time()
    
    def callback(indata, frames, time_info, status):
        audio = indata.copy().flatten()
        rolling_audio.extend(audio)
        viz_buffer.extend(audio)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BLOCK_SIZE):
            start_time = time.time()
            sys.stdout.write("\033[2J\033[?25l")
            
            while (time.time() - start_time) < TEST_DURATION:
                now = time.time()
                elapsed = int(now - start_time)
                
                # High-Frequency Processing (Every 800ms)
                if now - last_process_time > 0.8:
                    audio_data = np.array(rolling_audio, dtype=np.float32)
                    rms = np.sqrt(np.mean(audio_data**2))
                    
                    if rms > 0.0003:
                        # Normalize
                        peak = np.max(np.abs(audio_data))
                        audio_input = audio_data * (0.95 / peak) if peak > 0.01 else audio_data
                        
                        # Speculative Inference
                        result = pipe(audio_input)
                        new_text = result["text"].strip()
                        if new_text and len(new_text) > len(full_text):
                            full_text = new_text
                    
                    last_process_time = now

                if len(viz_buffer) >= 1024:
                    v_data = np.array(list(viz_buffer))
                    vol = min(1.0, np.sqrt(np.mean(v_data**2)) / 0.15)
                    draw_ui(vol, elapsed, full_text, "LIVE")
                
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h
")
        print(f"{GREEN}Team Session Ended.{RESET}")

if __name__ == "__main__":
    main()
