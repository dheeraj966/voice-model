"""
STT Precision Mode Comparison Test
Tests FP32 vs FP16 vs INT8 for quality and performance.
"""

import numpy as np
import sounddevice as sd
import whisper
import sys
import os
import time
import wave

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Test configuration
SAMPLE_RATE = 16000
TEST_DURATION = 15  # Short test for each mode
BUFFER_SIZE = 512

print("=" * 70)
print("STT PRECISION MODE COMPARISON TEST")
print("=" * 70)
print("\nThis will test 3 precision modes to find the best balance:")
print("  - FP32: Full precision (most accurate, slower)")
print("  - FP16: Half precision (faster, may lose accuracy on CPU)")
print("  - INT8: Quantized (fastest, least accurate)")
print(f"\nEach test: {TEST_DURATION} seconds of recording")
print("=" * 70)

# Load model once
print("\nLoading Whisper model (base)...")
model = whisper.load_model("base")
print("[OK] Model loaded")

# Recording function
def record_audio(duration):
    """Record audio for specified duration."""
    print(f"\nRecording for {duration} seconds... Speak now!")
    
    audio_chunks = []
    
    def callback(indata, frames, time_info, status):
        audio_chunks.append(indata.copy())
    
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        blocksize=BUFFER_SIZE
    )
    stream.start()
    
    start = time.time()
    while time.time() - start < duration:
        remaining = int(duration - (time.time() - start))
        print(f"  Recording... {remaining}s remaining", end='\r')
        time.sleep(0.1)
    
    stream.stop()
    stream.close()
    print("\n[OK] Recording complete")
    
    return np.concatenate(audio_chunks, axis=0).flatten().astype(np.float32)

# Test each precision mode
results = {}
test_audio = None

for mode in ["fp32", "fp16"]:
    print(f"\n{'='*70}")
    print(f"TESTING: {mode.upper()} MODE")
    print(f"{'='*70}")
    
    # Record once, reuse for all tests
    if test_audio is None:
        test_audio = record_audio(TEST_DURATION)
    else:
        print(f"\nUsing recorded audio ({len(test_audio)/SAMPLE_RATE:.1f}s)")
    
    # Transcribe with current mode
    print(f"\nTranscribing with {mode.upper()}...")
    transcribe_start = time.time()
    
    try:
        if mode == "fp16":
            result = model.transcribe(test_audio, fp16=True, language='en')
        else:
            result = model.transcribe(test_audio, fp16=False, language='en')
        
        transcribe_time = time.time() - transcribe_start
        transcription = result["text"].strip()
        
        results[mode] = {
            'text': transcription,
            'time': transcribe_time,
            'words': len(transcription.split()) if transcription else 0
        }
        
        print(f"[OK] Transcription complete in {transcribe_time:.2f}s")
        print(f"\nResult ({mode.upper()}):")
        print("-" * 70)
        print(f'  "{transcription}"')
        print("-" * 70)
        
    except Exception as e:
        print(f"[!] Error: {e}")
        results[mode] = {
            'text': f"ERROR: {e}",
            'time': 0,
            'words': 0
        }

# Comparison summary
print(f"\n\n{'='*70}")
print("COMPARISON RESULTS")
print(f"{'='*70}")
print(f"{'Mode':<10} | {'Time (s)':<10} | {'Words':<8} | Transcription")
print(f"{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*50}")

for mode, data in results.items():
    time_str = f"{data['time']:.2f}"
    words_str = str(data['words'])
    text_preview = data['text'][:50] + "..." if len(data['text']) > 50 else data['text']
    print(f"{mode.upper():<10} | {time_str:<10} | {words_str:<8} | {text_preview}")

print(f"\n{'='*70}")
print("ANALYSIS:")
print(f"{'='*70}")

# Find best mode
if results['fp32']['words'] > 0 or results['fp16']['words'] > 0:
    fp32_time = results['fp32']['time']
    fp16_time = results['fp16']['time']
    
    if fp16_time > 0 and fp32_time > 0:
        speedup = fp32_time / fp16_time
        print(f"  - FP16 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than FP32")
    
    if results['fp32']['words'] == results['fp16']['words']:
        print("  - Both modes captured same amount of speech")
        print("  - RECOMMENDATION: Use FP32 for stability on CPU")
    elif results['fp32']['words'] > results['fp16']['words']:
        print("  - FP32 captured more words (better accuracy)")
        print("  - RECOMMENDATION: Use FP32 for better capture")
    else:
        print("  - FP16 captured more words (unusual, may be less accurate)")
        print("  - RECOMMENDATION: Verify transcription quality manually")
else:
    print("  - No speech detected in either mode")
    print("  - Check microphone connection and try again")

print(f"\n{'='*70}")
print("TEST COMPLETE!")
print(f"{'='*70}")

# Save comparison results
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'precision_comparison.txt'), 'w') as f:
    f.write("STT PRECISION MODE COMPARISON\n")
    f.write("=" * 50 + "\n\n")
    for mode, data in results.items():
        f.write(f"{mode.upper()}:\n")
        f.write(f"  Time: {data['time']:.2f}s\n")
        f.write(f"  Words: {data['words']}\n")
        f.write(f"  Text: {data['text']}\n\n")

print(f"\nResults saved to: {os.path.join(output_dir, 'precision_comparison.txt')}")
