"""
Real-time Audio Visualizer / Equalizer
Shows live microphone input with volume bars and frequency spectrum.
Use this to test how naturally the STT captures your voice.
"""

import numpy as np
import sounddevice as sd
import sys
import os
import time
from collections import deque

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Settings
SAMPLE_RATE = 44100
FFT_SIZE = 1024
BAR_COUNT = 16
WINDOW_DURATION = 0.1  # 100ms update rate

print("=" * 60)
print("REAL-TIME AUDIO VISUALIZER / EQUALIZER")
print("=" * 60)
print("\nThis shows your microphone input in real-time.")
print("Speak naturally to see how well the STT captures your voice.")
print("\nPress Ctrl+C to stop.\n")
print("=" * 60)

# Buffer for audio data
audio_buffer = deque(maxlen=int(SAMPLE_RATE * WINDOW_DURATION))

# Volume history for smoothing
volume_history = deque(maxlen=5)

def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream."""
    if status:
        print(f"Status: {status}")
    audio_buffer.extend(indata.copy())

def calculate_bars(audio_data):
    """Calculate frequency bars from audio data."""
    if len(audio_data) < FFT_SIZE:
        return [0] * BAR_COUNT
    
    # Apply Hanning window
    window = np.hanning(len(audio_data))
    windowed = audio_data * window
    
    # FFT
    fft = np.abs(np.fft.rfft(windowed))
    
    # Group into bars (logarithmic spacing)
    bars = []
    freq_per_bar = len(fft) // BAR_COUNT
    for i in range(BAR_COUNT):
        start = i * freq_per_bar
        end = start + freq_per_bar
        if end <= len(fft):
            bar_value = np.mean(fft[start:end])
            bars.append(bar_value)
    
    # Normalize
    max_val = max(bars) if max(bars) > 0 else 1
    bars = [min(1.0, v / max_val) for v in bars]
    
    return bars

def draw_visualizer(bars, volume):
    """Draw the equalizer in terminal."""
    # Clear screen (works on most terminals)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=" * 60)
    print("REAL-TIME AUDIO VISUALIZER")
    print("=" * 60)
    print()
    
    # Volume meter
    vol_percent = int(volume * 100)
    vol_bar = "█" * int(volume * 40)
    vol_empty = "░" * (40 - len(vol_bar))
    print(f"VOLUME: [{vol_bar}{vol_empty}] {vol_percent:3d}%")
    print()
    
    # Frequency bars (equalizer)
    print("FREQUENCY SPECTRUM (Low ← → High):")
    print("-" * 60)
    
    bar_height = 12
    for row in range(bar_height, 0, -1):
        line = "  "
        for bar in bars:
            bar_fill = int(bar * bar_height)
            if bar_fill >= row:
                line += "██ "
            elif bar_fill > row - 0.5:
                line += "▄▄ "
            else:
                line += "   "
        print(line)
    
    print("  " + "─" * (BAR_COUNT * 3))
    print("   LOW                    MID                   HIGH")
    print()
    
    # Volume level indicator
    if volume > 0.7:
        level = "LOUD"
        indicator = "🔊"
    elif volume > 0.3:
        level = "MODERATE"
        indicator = "🔉"
    elif volume > 0.05:
        level = "QUIET"
        indicator = "🔈"
    else:
        level = "SILENCE"
        indicator = "🔇"
    
    print(f"Audio Level: {indicator} {level}")
    print("=" * 60)
    print("Speak naturally to test STT input quality")
    print("Press Ctrl+C to stop")

def main():
    """Main function."""
    try:
        # Start audio stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=FFT_SIZE
        )
        stream.start()
        
        print("Starting audio capture...")
        time.sleep(0.5)
        
        while True:
            if len(audio_buffer) >= FFT_SIZE:
                # Get recent audio
                audio_data = np.array(audio_buffer)[-FFT_SIZE:]
                
                # Calculate volume (RMS)
                rms = np.sqrt(np.mean(audio_data ** 2))
                volume = min(1.0, rms / 0.1)  # Normalize to 0.1 max
                
                # Smooth volume
                volume_history.append(volume)
                avg_volume = np.mean(volume_history)
                
                # Calculate frequency bars
                bars = calculate_bars(audio_data)
                
                # Draw visualization
                draw_visualizer(bars, avg_volume)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'stream' in locals():
            stream.stop()
            stream.close()

if __name__ == "__main__":
    main()
