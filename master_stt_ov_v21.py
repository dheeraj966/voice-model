"""
MASTER STT OV V21 - INTEL IRIS XE ULTRA-REALTIME
- Architecture: OpenVINO GPU Decoupled (V21) - Native Generate Implementation
- Feature: High-Load Nuance Capturing (GPU 50%+)
- Logic: Live Snippet Processing vs. Post-Session Total Log
- Diagnostics: Process Load Monitoring
"""

import numpy as np
import sounddevice as sd
import sys
import os
import time
import wave
import psutil
import torch
from collections import deque
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

# Windows-Specific Terminal Fixes
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# Global Constants
SAMPLE_RATE = 16000
BLOCK_SIZE = 4096       
TEST_DURATION = 300     
MODEL_ID = "openai/whisper-medium.en"  
VAD_RMS_THRESHOLD = 0.0004             
BEAM_SIZE = 5                          
TRANSCRIPTION_INTERVAL = 1.0           

# ANSI Styles
CLEAR = "\033[H\033[J"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

def get_process_load():
    return psutil.cpu_percent(), psutil.virtual_memory().percent

def draw_ui(vol, elapsed, text, status, cpu_load, mem_load):
    sys.stdout.write("\033[H")
    print(f"{BLUE}╔{'═'*68}╗{RESET}")
    print(f"{BLUE}║{RESET} {BOLD}MASTER STT V21 (IRIS XE GPU){RESET} | {elapsed//60:02d}m {elapsed%60:02d}s | Status: {status:<10} {BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    v_width = int(vol * 40)
    meter = (GREEN if vol < 0.3 else YELLOW) + "█" * v_width + RESET + "░" * (40 - v_width)
    print(f"{BLUE}║{RESET} MIC SENSITIVITY: [{meter}] {int(vol*100):3d}% {' '*2}{BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    print(f"{BLUE}║{RESET} {BOLD}DIAGNOSTICS:{RESET} CPU: {cpu_load:4.1f}% | RAM: {mem_load:4.1f}% | GPU: {CYAN}MAX-THROUGHPUT{RESET} {' '*5}{BLUE}║{RESET}")
    print(f"{BLUE}╠{'═'*68}╣{RESET}")
    print(f"{BLUE}║{RESET} {CYAN}REAL-TIME NUANCE:{RESET}{' '*50}{BLUE}║{RESET}")
    display_text = text[-130:] if len(text) > 130 else text
    line1, line2 = display_text[:64], display_text[64:128]
    print(f"{BLUE}║{RESET}  > {line1:<64} {BLUE}║{RESET}")
    print(f"{BLUE}║{RESET}    {line2:<64} {BLUE}║{RESET}")
    print(f"{BLUE}╚{'═'*68}╝{RESET}")

def main():
    print(f"{YELLOW}Waking up Intel Iris Xe GPU Engine ({MODEL_ID})...{RESET}")
    print(f"{CYAN}[1/4] Loading and Compiling Model for GPU High-Load...{RESET}")
    
    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, 
            device="GPU",           
            ov_config={"CACHE_DIR": "ov_cache", "PERFORMANCE_HINT": "THROUGHPUT"}, 
            export=True             
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print(f"{GREEN}[2/4] GPU Compilation Successful.{RESET}")

        def transcribe_audio(audio_data, beams=BEAM_SIZE, max_tokens=128):
            inputs = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features
            generated_ids = model.generate(input_features, num_beams=beams, max_new_tokens=max_tokens)
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"{CYAN}[3/4] Pre-Warming GPU...{RESET}")
        transcribe_audio(np.zeros(16000, dtype=np.float32))
        print(f"{GREEN}[4/4] AI Ready (Iris Xe - Native Implementation).{RESET}")
    except Exception as e:
        print(f"{RED}OPEN-VINO LOAD ERROR: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return

    full_audio_log = []
    viz_buffer = deque(maxlen=2048)
    rolling_audio = deque(maxlen=int(SAMPLE_RATE * 5.0))
    current_transcription = ""
    status = f"{CYAN}LISTENING{RESET}"
    last_trans_time = time.time()
    
    def callback(indata, frames, time_info, status_msg):
        audio = indata.copy().flatten()
        viz_buffer.extend(audio)
        rolling_audio.extend(audio)
        full_audio_log.append(audio)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BLOCK_SIZE):
            start_time = time.time()
            sys.stdout.write(CLEAR + "\033[?25l")
            while (time.time() - start_time) < TEST_DURATION:
                now = time.time()
                elapsed = int(now - start_time)
                if now - last_trans_time > TRANSCRIPTION_INTERVAL:
                    status = f"{YELLOW}THINKING{RESET}"
                    audio_raw = np.array(rolling_audio, dtype=np.float32)
                    rms = np.sqrt(np.mean(audio_raw**2))
                    if rms > VAD_RMS_THRESHOLD:
                        peak = np.max(np.abs(audio_raw))
                        audio_processed = audio_raw * (0.98 / peak) if peak > 0 else audio_raw
                        current_transcription = transcribe_audio(audio_processed)
                    last_trans_time = now
                    status = f"{CYAN}LISTENING{RESET}"
                if len(viz_buffer) >= 1024:
                    v_data = np.array(list(viz_buffer))
                    vol = min(1.0, np.sqrt(np.mean(v_data**2)) / 0.15)
                    cpu_l, mem_l = get_process_load()
                    draw_ui(vol, elapsed, current_transcription, status, cpu_l, mem_l)
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")
        print(f"\n{GREEN}LIVE SESSION ENDED.{RESET}")
        if full_audio_log:
            print(f"{YELLOW}Generating Final Summary...{RESET}")
            final_audio = np.concatenate(full_audio_log)
            os.makedirs("output", exist_ok=True)
            out_file = f"output/v21_gpu_{int(time.time())}.wav"
            with wave.open(out_file, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes((final_audio * 32767).astype(np.int16).tobytes())
            final_text = transcribe_audio(final_audio.astype(np.float32), beams=BEAM_SIZE, max_tokens=448)
            print(f"\n{BLUE}{'═'*70}{RESET}")
            print(f"{GREEN}{BOLD}FINAL SESSION SUMMARY:{RESET}")
            print(f"\n\"{final_text if final_text else '(No speech detected)'}\"\n")
            print(f"{BLUE}{'═'*70}{RESET}")
            print(f"File: {out_file}\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
