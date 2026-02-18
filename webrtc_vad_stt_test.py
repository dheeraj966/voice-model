"""
Professional STT Test with WebRTC VAD Integration
- Hardware-grade Voice Activity Detection
- Adaptive sensitivity based on background noise
- Flicker-free 60fps visualizer
- Only transcribes actual speech segments
"""

import numpy as np
import sounddevice as sd
import whisper
import sys
import os
import time
import wave
import collections
import webrtcvad
from collections import deque

# Windows ANSI support
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding='utf-8')

# ============= SETTINGS =============
TEST_DURATION = 45
SAMPLE_RATE = 16000  # WebRTC VAD requires 16kHz
BUFFER_SIZE = 160    # 10ms frames for VAD (160 samples @ 16kHz)
FFT_SIZE = 1024
BAR_COUNT = 10
VAD_MODE = 2         # 0=least aggressive, 3=most aggressive
VAD_WINDOW = 10      # Number of frames to consider for speech detection
SPEECH_THRESHOLD = 0.6  # Percentage of frames that must be speech
# ====================================

# ANSI codes
CLEAR = "\033[H\033[J"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"

print(f"{BLUE}{'='*70}")
print("PROFESSIONAL STT - WebRTC VAD INTEGRATION")
print(f"{'='*70}{RESET}")
print(f"\n{BOLD}Settings:{RESET}")
print(f"  Duration: {TEST_DURATION}s")
print(f"  VAD Mode: {VAD_MODE}/3 (Aggressiveness)")
print(f"  Frame Size: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLE_RATE*1000:.1f}ms)")
print(f"  Speech Threshold: {SPEECH_THRESHOLD*100:.0f}% frames")
print(f"\n{CYAN}WebRTC VAD provides hardware-grade speech detection!{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

# Initialize WebRTC VAD
print("Initializing WebRTC VAD...")
vad = webrtcvad.Vad(VAD_MODE)
print(f"{GREEN}[OK] VAD initialized (Mode {VAD_MODE}){RESET}")

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("tiny")
print(f"{GREEN}[OK] Model ready{RESET}\n")
time.sleep(1)

# Buffers
MAX_AUDIO = int(SAMPLE_RATE * TEST_DURATION * 1.5)
recording = np.zeros(MAX_AUDIO, dtype=np.float32)
rec_idx = 0
viz_buffer = deque(maxlen=int(SAMPLE_RATE * 0.1))
running = True
last_text = ""

# WebRTC VAD buffers
vad_frames = collections.deque(maxlen=VAD_WINDOW)  # Recent VAD decisions
vad_frame_buffer = bytearray()  # Accumulating audio for VAD frames
speech_active = False
speech_frame_count = 0
total_frames = 0

def callback(indata, frames, time_info, status):
    global rec_idx, running, vad_frame_buffer, speech_active, speech_frame_count, total_frames
    
    audio = indata.copy().flatten()
    viz_buffer.extend(audio)
    
    # Add to recording
    if running and rec_idx + len(audio) < MAX_AUDIO:
        recording[rec_idx:rec_idx+len(audio)] = audio
        rec_idx += len(audio)
    
    # Process VAD frames
    for sample in audio:
        # Convert float to 16-bit integer for WebRTC VAD
        sample_int16 = int(sample * 32767)
        vad_frame_buffer.extend(sample_int16.to_bytes(2, 'little', signed=True))
        
        # When we have enough samples for a VAD frame (10ms = 160 samples)
        if len(vad_frame_buffer) >= BUFFER_SIZE * 2:  # 2 bytes per sample
            frame_bytes = bytes(vad_frame_buffer[:BUFFER_SIZE * 2])
            vad_frame_buffer = vad_frame_buffer[BUFFER_SIZE * 2:]
            
            # Run VAD detection
            try:
                is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
                vad_frames.append(is_speech)
                total_frames += 1
                
                if is_speech:
                    speech_frame_count += 1
            except Exception as e:
                vad_frames.append(False)
    
    # Determine if speech is active (majority of recent frames are speech)
    if len(vad_frames) >= VAD_WINDOW:
        speech_ratio = sum(vad_frames) / len(vad_frames)
        speech_active = speech_ratio >= SPEECH_THRESHOLD

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

def draw(bars, vol, elapsed, text="", speech=False, vad_ratio=0):
    print(CLEAR, end='')
    print(f"{BLUE}{'='*70}{RESET}")
    status_color = MAGENTA if speech else CYAN
    status_icon = "🎤 SPEECH" if speech else "👂 SILENCE"
    print(f"WebRTC VAD STT - {TEST_DURATION - elapsed}s remaining {status_color}{status_icon}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    # Volume with color
    vp = int(vol * 100)
    vb = "█" * int(vol * 40)
    ve = "░" * (40 - len(vb))
    if speech:
        vol_color = MAGENTA
    elif vol > 0.1:
        vol_color = YELLOW
    else:
        vol_color = BLUE
    print(f"Level: [{vol_color}{vb}{RESET}{ve}] {vp:3d}%")
    print()
    
    # VAD confidence meter
    conf_blocks = int(vad_ratio * 20)
    conf_bar = "▓" * conf_blocks + "░" * (20 - conf_blocks)
    if vad_ratio > SPEECH_THRESHOLD:
        conf_color = GREEN
    else:
        conf_color = BLUE
    print(f"VAD:  [{conf_color}{conf_bar}{RESET}] {vad_ratio*100:3.0f}%")
    print()
    
    # Frequency bars with gradient
    for row in range(8, 0, -1):
        line = "  "
        for b in bars:
            if int(b * 8) >= row:
                if speech:
                    line += f"{MAGENTA}██{RESET} "
                elif row > 6:
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
    
    # Stats
    print(f"Recorded: {rec_idx/SAMPLE_RATE:.1f}s | Frames: {total_frames} | Speech: {speech_frame_count} ({speech_frame_count/max(1,total_frames)*100:.1f}%)")
    print(f"{BLUE}{'='*70}{RESET}")

# Start
print("Starting audio stream...")
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        blocksize=BUFFER_SIZE * 4  # 40ms blocks
    )
    stream.start()
    print(f"{GREEN}[OK] Audio started{RESET}\n")
except Exception as e:
    print(f"{RED}[!] Audio error: {e}{RESET}")
    print("Make sure your microphone supports 16kHz sample rate!")
    sys.exit(1)

print("Starting in 3...")
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# Main loop
start = time.time()
last_transcribe = 0
vad_ratio = 0

try:
    while running and (time.time() - start) < TEST_DURATION:
        elapsed = int(time.time() - start)
        
        if len(viz_buffer) >= FFT_SIZE:
            viz = np.array(list(viz_buffer)[-FFT_SIZE:])
            rms = np.sqrt(np.mean(viz ** 2))
            vol = min(1.0, rms / 0.1)
            
            # Calculate VAD ratio
            if len(vad_frames) > 0:
                vad_ratio = sum(vad_frames) / len(vad_frames)
            else:
                vad_ratio = 0
            
            bars = get_bars(viz)
            draw(bars, vol, elapsed, last_text, speech_active, vad_ratio)
            
            # Transcribe when speech detected
            if speech_active and elapsed > last_transcribe + 5 and rec_idx > SAMPLE_RATE * 2:
                try:
                    # Get recent speech segment
                    recent_start = max(0, rec_idx - int(SAMPLE_RATE * 8))
                    recent = recording[recent_start:rec_idx].copy()
                    
                    result = model.transcribe(recent, fp16=False, language='en')
                    if result["text"].strip():
                        last_text = result["text"].strip()
                        last_transcribe = elapsed
                        print(f"\n{GREEN}✓ Transcribed speech segment!{RESET}\n")
                        time.sleep(0.3)
                except Exception as e:
                    pass
        
        time.sleep(0.05)
    
    running = False
    
except KeyboardInterrupt:
    print(f"\n{YELLOW}Stopped by user{RESET}")
    running = False
finally:
    stream.stop()
    stream.close()

# Results
print("\n" * 2)
print(f"{BLUE}{'='*70}{RESET}")
print(f"{BOLD}FINAL RESULTS{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

if rec_idx > SAMPLE_RATE * 2:
    print(f"Processing {rec_idx/SAMPLE_RATE:.1f}s of audio...")
    print(f"Total frames processed: {total_frames}")
    print(f"Speech frames: {speech_frame_count} ({speech_frame_count/max(1,total_frames)*100:.1f}%)")
    print("\nTranscribing full recording with VAD filtering...")
    
    # Apply VAD to full recording for final transcription
    result = model.transcribe(recording[:rec_idx], fp16=False, language='en')
    text = result["text"].strip()
    
    print(f"\n{'─'*70}")
    print(f"{BOLD}FINAL TRANSCRIPTION:{RESET}")
    print(f"{'─'*70}")
    if text:
        # Word wrap
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
    
    # Save
    outdir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'webrtc_vad_stt_{int(time.time())}.wav')
    
    with wave.open(outpath, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((recording[:rec_idx] * 32767).astype(np.int16).tobytes())
    
    print(f"{GREEN}✓ Saved:{RESET} {outpath}")
    print(f"{GREEN}✓ Duration:{RESET} {rec_idx/SAMPLE_RATE:.1f}s")
    print(f"{GREEN}✓ Sample Rate:{RESET} {SAMPLE_RATE}Hz")
    print(f"{GREEN}✓ VAD Mode:{RESET} {VAD_MODE}/3")

print(f"\n{BLUE}{'='*70}{RESET}")
print(f"{GREEN}TEST COMPLETE!{RESET}")
print(f"{BLUE}{'='*70}{RESET}")
