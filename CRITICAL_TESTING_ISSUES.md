# CRITICAL TESTING ISSUES - For Gemini Agent

## Context
Another agent is modifying the code. Below are the **CRITICAL ONLY** testing issues that must be fixed.

---

## 🔴 CRITICAL ISSUE #1: Thread Safety Violations (Race Conditions)

**Files**: `extended_stt_test.py`, `optimized_stt_test.py`

**Problem**: Shared state accessed by multiple threads without locks
```python
# Line 45-48 in extended_stt_test.py
recorded_chunks = []  # ❌ Modified by audio_callback AND transcribe_worker
current_transcription = ""  # ❌ Modified by worker, read by main thread
```

**Impact**: Crashes at exactly 5 seconds due to data corruption

**Fix Required**:
```python
from threading import Lock
chunks_lock = Lock()
transcription_lock = Lock()

def audio_callback(indata, frames, time_info, status):
    with chunks_lock:
        recorded_chunks.append(indata.copy())

def transcribe_worker():
    with chunks_lock:
        audio_data = np.concatenate(recorded_chunks, axis=0)
    with transcription_lock:
        current_transcription = result["text"]
```

---

## 🔴 CRITICAL ISSUE #2: Memory Leak - Unbounded Buffer Growth

**Files**: `extended_stt_test.py`, `optimized_stt_test.py`

**Problem**: `recorded_chunks` list grows indefinitely for entire test duration
```python
recorded_chunks = []  # ❌ UNBOUNDED - 60 seconds = ~60MB
```

**Impact**: Memory pressure → GC pauses → audio dropouts → crashes

**Fix Required**:
```python
from collections import deque
# Keep only last 10 seconds
recorded_chunks = deque(maxlen=int(SAMPLE_RATE * 10 / BLOCK_SIZE))
```

---

## 🔴 CRITICAL ISSUE #3: Blocking Operations Without Timeouts

**Files**: `test_components.py`, `multiprocessing_stt_test.py`, `extended_stt_test.py`

**Problem**: Transcription can hang indefinitely
```python
result = model.transcribe(audio_data, fp16=False)  # ❌ No timeout, can hang 10+ seconds
```

**Impact**: Test hangs, worker falls behind, audio buffer overflows

**Fix Required**:
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Transcription timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)  # 10-second timeout
try:
    result = model.transcribe(...)
finally:
    signal.alarm(0)
```

---

## 🔴 CRITICAL ISSUE #4: Audio Clipping in WAV Export

**Files**: `test_stt.py`, `precision_comparison_test.py`

**Problem**: No normalization before int16 conversion
```python
audio_int16 = (audio * 32767).astype(np.int16)  # ❌ Clips if audio > 1.0
```

**Impact**: Distorted audio files, inconsistent transcription results

**Fix Required**:
```python
audio_normalized = np.clip(audio * 0.9, -1.0, 1.0)  # Prevent clipping
audio_int16 = (audio_normalized * 32767).astype(np.int16)
```

---

## 🔴 CRITICAL ISSUE #5: Silent Error Failures

**Files**: All test files

**Problem**: Errors logged but tests continue in broken state
```python
except Exception as e:
    error_log.append(f"Error: {str(e)[:40]}")  # ❌ Silent failure
    # Test continues with invalid state
```

**Impact**: False positives, corrupted results

**Fix Required**:
```python
except Exception as e:
    print(f"[FAIL] Critical error: {e}")
    recording_active = False  # Stop test immediately
    return False  # Mark test as failed
```

---

## 🟡 MODERATE ISSUE #6: Inconsistent Model Loading

**Files**: All test files load models independently

**Problem**: Each test loads its own model instance
```python
# test_stt.py: model = whisper.load_model("tiny")
# optimized_stt_test.py: model = whisper.load_model("base")  # Different!
```

**Impact**: Wasted time (30-60s per test), inconsistent results

**Fix Required**: Centralize model loading in shared fixture or conftest.py

---

## 🟡 MODERATE ISSUE #7: Missing Test Assertions

**Files**: `test_stt.py`, `extended_stt_test.py`, `optimized_stt_test.py`

**Problem**: Tests print results but never validate them
```python
print(f"You said: \"{transcription.strip()}\"")  # ❌ No assertion
# Test always passes regardless of output
```

**Fix Required**:
```python
assert len(transcription.strip()) > 0, "Transcription should not be empty"
assert result["language"] == "en", "Language detection failed"
```

---

## Priority Order for Fixes

1. **Thread safety locks** (prevents 5s crashes)
2. **Bounded buffers** (prevents memory leaks)
3. **Timeouts** (prevents hangs)
4. **Audio normalization** (prevents distortion)
5. **Error handling** (stops broken tests)
6. **Model centralization** (optimization)
7. **Assertions** (makes tests actually validate)

---

## Files Requiring Critical Fixes

| File | Issues | Priority |
|------|--------|----------|
| `extended_stt_test.py` | #1, #2, #3, #5 | 🔴 CRITICAL |
| `optimized_stt_test.py` | #1, #2, #3, #5 | 🔴 CRITICAL |
| `test_stt.py` | #4, #5, #7 | 🔴 CRITICAL |
| `test_components.py` | #3, #5, #6 | 🔴 CRITICAL |
| `multiprocessing_stt_test.py` | #3, #5 | 🔴 CRITICAL |
| `precision_comparison_test.py` | #4, #5 | 🔴 CRITICAL |

---

## DO NOT MODIFY

- `whisper stt/` directory (OpenAI Whisper source)
- `KittenTTS/` directory (third-party TTS)
- `DeepSeek-R1/` directory (documentation only)

---

## Testing After Fixes

Run these commands to verify fixes:
```bash
# Quick stability test (should complete 60s without crash)
python ultra_stable_stt.py

# Component tests
python test_components.py

# Precision test
python precision_comparison_test.py
```

All tests should:
- ✅ Complete full duration without crashes
- ✅ Show consistent transcription results
- ✅ Have proper error handling
- ✅ Clean up resources properly

---

**Generated**: 2026-02-19
**For**: Gemini Agent code modifications
**Scope**: CRITICAL testing issues only
