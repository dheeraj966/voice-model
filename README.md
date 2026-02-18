# Voice Agent - STT Component

Speech-to-Text (STT) implementation using OpenAI Whisper with optimized real-time audio capture and visualization.

## 🎯 Features

- **Real-time Audio Capture**: 16kHz sample rate with PortAudio-safe buffering
- **Live Visualization**: Frequency spectrum analyzer with volume meter
- **Multiprocessing Architecture**: Separate AI inference process (no UI blocking)
- **WebRTC VAD**: Voice Activity Detection for noise rejection
- **GPU Acceleration**: DirectML support for Intel Iris Xe (optional)

## 📁 Project Structure

```
voice-model/
├── whisper stt/          # OpenAI Whisper STT engine
├── *.py                  # STT test scripts
├── output/               # Generated audio files (gitignored)
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Core dependencies
pip install numpy sounddevice whisper faster-whisper webrtcvad

# Optional: GPU acceleration (Intel Iris Xe)
pip install onnxruntime-directml optimum-intel openvino
```

### Running Tests

#### **Recommended: Audio-Decoupled STT (Most Stable)**
```bash
python final_audio_decoupled_stt.py
```
- 512ms buffer (PortAudio-safe)
- Pre-loaded model (no cold start)
- Multiprocessing (zero UI blocking)
- **Duration**: 30-60 seconds

#### **With WebRTC VAD (Best Noise Rejection)**
```bash
python webrtc_vad_stt_test.py
```
- Hardware-grade voice detection
- Only transcribes actual speech
- **Duration**: 45 seconds

#### **GPU-Accelerated (Intel Iris Xe)**
```bash
gpu_accelerated_stt.py
```
- DirectML backend
- Double-engine architecture
- **Duration**: 60 seconds

#### **Ultra-Stable (CPU Only)**
```bash
python ultra_stable_stt.py
```
- Single-threaded (no threading issues)
- Guaranteed 60-second runtime
- Transcription at end only

## 🔧 Configuration

### Buffer Size (Critical for Stability)
```python
BUFFER_SIZE = 8192  # 512ms window (recommended)
# vs
BUFFER_SIZE = 1024  # 64ms window (may crash)
```

### Model Selection
```python
MODEL_NAME = "tiny.en"  # English-only, fastest
MODEL_NAME = "base.en"  # Better accuracy, slower
```

### Test Duration
```python
TEST_DURATION = 30  # Quick verification
TEST_DURATION = 60  # Full test
```

## 📊 Test Scripts Comparison

| Script | Duration | GPU | Live Transcription | Stability | Best For |
|--------|----------|-----|-------------------|-----------|----------|
| `final_audio_decoupled_stt.py` | 30-60s | ❌ | ✅ | ⭐⭐⭐⭐⭐ | **Production** |
| `webrtc_vad_stt_test.py` | 45s | ❌ | ✅ | ⭐⭐⭐⭐ | Noisy environments |
| `gpu_accelerated_stt.py` | 60s | ✅ | ✅ | ⭐⭐⭐⭐ | Intel Iris Xe |
| `ultra_stable_stt.py` | 60s | ❌ | ❌ | ⭐⭐⭐⭐⭐ | Guaranteed stability |
| `multiprocessing_stt_test.py` | 60s | ❌ | ✅ | ⭐⭐⭐⭐⭐ | Zero GIL blocking |

## 🏆 Key Fixes Implemented

### 5-Second Crash Solution

**Root Cause**: PortAudio buffer underrun when torch/MKL blocks CPU

**Fixes Applied**:
1. **Large Buffer**: 8192 samples (512ms) vs 1024 (64ms)
2. **Pre-loaded Model**: Load BEFORE audio starts
3. **Multiprocessing**: Separate AI process (GIL-bypassed)
4. **Non-blocking Queue**: `maxsize=1` (drops audio, never blocks)
5. **English-only Model**: `tiny.en` (20% faster)

## 📝 Requirements

```txt
numpy
sounddevice
openai-whisper
faster-whisper
webrtcvad
scipy
```

## 🎯 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN PROCESS                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • UI Rendering (20 FPS)                        │   │
│  │  • Audio Capture (sounddevice)                  │   │
│  │  • FFT Calculations                             │   │
│  │  • NEVER blocked by AI                          │   │
│  └─────────────────────────────────────────────────┘   │
│              │                        │                 │
│         Queue (audio)           Queue (results)        │
│              │                        │                 │
└──────────────┼────────────────────────┼─────────────────┘
               │                        │
               │                        │
               ▼                        │
┌─────────────────────────────────────────────────────────┐
│                   WORKER PROCESS                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  • Whisper Model (isolated memory)              │   │
│  │  • Heavy AI Inference                           │   │
│  │  • GIL-bypassed (separate Python interpreter)   │   │
│  │  • NEVER blocks UI                              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 🔮 Future Work

- [ ] LLM Integration (DeepSeek-R1)
- [ ] TTS Integration (KittenTTS)
- [ ] Full Pipeline: STT → LLM → TTS
- [ ] OpenVINO Model Optimization
- [ ] Streaming Transcription

## 📄 License

MIT License

## 🤝 Contributing

This is a work in progress. LLM and TTS components are not yet integrated.
