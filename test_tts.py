"""
Text-to-Speech (TTS) Test Script
Uses KittenTTS to generate speech from text.
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 50)
print("KITTENTTS - TEXT TO SPEECH TEST")
print("=" * 50)

# Test import
print("\nImporting KittenTTS...")
try:
    import kittentts
    print("[OK] KittenTTS imported successfully!")
except Exception as e:
    print(f"[FAIL] Could not import KittenTTS: {e}")
    sys.exit(1)

# Test basic generation
print("\nGenerating speech for: 'Hello, this is a test.'")
try:
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tts_test.wav')
    
    # Generate audio
    print("Loading TTS model...")
    from kittentts import KittenTTS
    
    tts = KittenTTS()
    
    print("Generating audio...")
    audio = tts.generate("Hello, this is a test.")
    
    # Save to file
    import numpy as np
    import wave
    
    print(f"Saving to {output_path}...")
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(24000)  # Common sample rate
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"\n[OK] Audio saved to: {output_path}")
    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {len(audio)/24000:.2f} seconds")
    
except Exception as e:
    print(f"\n[FAIL] Error during TTS generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("TTS TEST COMPLETE!")
print("=" * 50)
