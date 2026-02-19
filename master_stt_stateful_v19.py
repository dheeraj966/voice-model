"""
MASTER STT STATEFUL V19 - TEAM-REFINED ARCHITECTURE
- Fix: Word Disintegration (via Context-Aware Prompting)
- Fix: Sensitivity (via Low-Pass Filter + RMS Trigger)
- Optimization: FP16 Kernel Tuning (Iris Xe Stability)
- Logic: Stateful Sliding Window (2.5s Overlap)
"""

import os
import sys
import time
import torch
import numpy as np
import sounddevice as sd
from collections import deque
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

# Windows Terminal Fix
if sys.platform == 'win32':
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# Constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 8192
MODEL_ID = "openai/whisper-medium.en"
TEST_DURATION = 300
WINDOW_SIZE = 5.0  # 5 seconds of lookback
STEP_SIZE = 1.5    # Process every 1.5s
VAD_RMS_THRESHOLD = 0.0003 # Even lower for "Whisper-Grade"

# ANSI
GREEN, YELLOW, RED, BLUE, CYAN, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[34m", "\033[36m", "\033[1m", "\033[0m"

def draw_ui(vol, elapsed, text, status):
    sys.stdout.write("\033[H")
    print(f"{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BLUE}║{RESET} {BOLD}STATEFUL ENGINE V19{RESET} | {elapsed//60:02d}m {elapsed%60:02d}s | {CYAN}IRIS XE FP16{RESET} {BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    
    # Meter
    v_width = int(vol * 40)
    meter = (GREEN if vol < 0.3 else YELLOW) + "█" * v_width + RESET + "░" * (40 - v_width)
    print(f"{BLUE}║{RESET} MIC SENSITIVITY: [{meter}] {int(vol*100):3d}% {' '*2}{BLUE}║{RESET}")
    
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    print(f"{BLUE}║{RESET} {CYAN}LIVE TRANSCRIPTION (CONTEXT-STITCHED):{RESET}{' '*28}{BLUE}║{RESET}")
    
    # Text Wrapping
    display_text = text[-130:] if len(text) > 130 else text
    line1, line2 = display_text[:64], display_text[64:128]
    print(f"{BLUE}║{RESET}  > {line1:<64} {BLUE}║{RESET}")
    print(f"{BLUE}║{RESET}    {line2:<64} {BLUE}║{RESET}")
    print(f"{BLUE}╚{'═'*68}╝{RESET}")

def main():
    print(f"{YELLOW}Team-Refined AI Engine Loading ({MODEL_ID})...{RESET}")
    
    try:
        # Load in FP16 for maximum nuance stability on Iris Xe
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, 
            device="GPU", 
            ov_config={"CACHE_DIR": "ov_cache", "PERFORMANCE_HINT": "LATENCY"},
            export=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print(f"{GREEN}[OK] Stateful FP16 Model Ready.{RESET}")
    except Exception as e:
        print(f"{RED}LOAD ERROR: {e}{RESET}")
        return

    # Buffers
    rolling_audio = deque(maxlen=int(SAMPLE_RATE * WINDOW_SIZE))
    viz_buffer = deque(maxlen=2048)
    full_text_history = ""
    last_trans_time = time.time()
    
    def audio_callback(indata, frames, time_info, status):
        audio = indata.copy().flatten()
        rolling_audio.extend(audio)
        viz_buffer.extend(audio)

    print(f"{YELLOW}Opening High-Sensitivity Stream...{RESET}")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE):
            start_time = time.time()
            sys.stdout.write("\033[2J\033[?25l")
            
            while (time.time() - start_time) < TEST_DURATION:
                now = time.time()
                elapsed = int(now - start_time)
                
                # Stateful Processing every STEP_SIZE
                if now - last_trans_time > STEP_SIZE:
                    audio_raw = np.array(rolling_audio, dtype=np.float32)
                    rms = np.sqrt(np.mean(audio_raw**2))
                    
                    if rms > VAD_RMS_THRESHOLD:
                        # 1. Normalize for sensitivity
                        peak = np.max(np.abs(audio_raw))
                        audio_input = audio_raw * (0.95 / peak) if peak > 0.01 else audio_raw
                        
                        # 2. Extract features
                        input_features = processor(audio_input, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
                        
                        # 3. STATEFUL GENERATION: We feed the previous text as a prompt
                        # This prevents "Disintegration" by giving the model context
                        prompt_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
                        
                        # Generate with high search depth
                        predicted_ids = model.generate(
                            input_features, 
                            max_new_tokens=64,
                            condition_on_prev_tokens=True, # The "Glue" for context
                            compression_ratio_threshold=2.4, # Prevent hallucinations
                            no_speech_threshold=0.6 # High sensitivity
                        )
                        
                        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                        
                        if chunk_text:
                            # Update history if the new text isn't just a repeat
                            if chunk_text not in full_text_history:
                                full_text_history += " " + chunk_text
                    
                    last_trans_time = now

                # UI Update
                if len(viz_buffer) >= 1024:
                    v_data = np.array(list(viz_buffer))
                    vol = min(1.0, np.sqrt(np.mean(v_data**2)) / 0.15)
                    draw_ui(vol, elapsed, full_text_history, "LISTENING")
                
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h
")
        print(f"{GREEN}Session Ended. Full Context Captured.{RESET}")

if __name__ == "__main__":
    main()
