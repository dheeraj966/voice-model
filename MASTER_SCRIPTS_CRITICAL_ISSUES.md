# CRITICAL ISSUES IN MASTER STT SCRIPTS - For Gemini Agent

## Context
Analysis of all `master_stt_*.py` files reveals **CRITICAL TESTING ISSUES** that must be fixed.

---

## 🔴 CRITICAL ISSUE #1: UNBOUNDED MEMORY GROWTH (All Master Scripts)

**Files**: ALL master_stt_*.py files
- `master_stt_fix.py` (Line 93)
- `master_stt_fix_ov.py` (Line 107)
- `master_stt_stateful_v19.py` (Line 94)
- `master_stt_policy_v18.py` (Line 122)
- `master_stt_distil_v20.py` (Line 95)

**Problem**: `full_audio_log` list grows indefinitely
```python
# Line 93 in master_stt_fix.py
full_audio_log = []  # ❌ UNBOUNDED - 5 minutes = 150MB+

def callback(indata, frames, time_info, status_msg):
    full_audio_log.append(audio)  # ❌ Appends every callback for 5 minutes
```

**Impact**:
- **150-300MB** memory accumulation over 5-minute test
- **GC pauses** cause audio dropouts
- **Eventual crash** on low-memory systems

**Fix Required**:
```python
from collections import deque
# Keep only last 30 seconds for final transcription
full_audio_log = deque(maxlen=int(SAMPLE_RATE * 30 / BLOCK_SIZE))

# OR limit to specific duration
full_audio_log = deque(maxlen=1000)  # Last 1000 chunks only
```

---

## 🔴 CRITICAL ISSUE #2: NO THREAD LOCKS ON SHARED STATE

**Files**: `master_stt_fix.py`, `master_stt_fix_ov.py`, `master_stt_stateful_v19.py`

**Problem**: Audio callback and main thread share variables without synchronization
```python
# Shared state
rolling_audio = deque(maxlen=int(SAMPLE_RATE * 5.0))
current_transcription = ""

# Audio callback (runs in SEPARATE THREAD)
def callback(indata, frames, time_info, status_msg):
    rolling_audio.extend(audio)  # ❌ No lock

# Main thread (reads same variable)
audio_to_transcribe = np.array(rolling_audio, dtype=np.float32)  # ❌ No lock
```

**Impact**:
- **RuntimeError**: deque mutated during iteration
- **Data corruption**: Partial reads of audio data
- **Crashes** at unpredictable times

**Fix Required**:
```python
from threading import Lock
audio_lock = Lock()

def callback(indata, frames, time_info, status_msg):
    with audio_lock:
        rolling_audio.extend(audio)

# In main loop:
with audio_lock:
    audio_to_transcribe = np.array(rolling_audio, dtype=np.float32).copy()
```

---

## 🔴 CRITICAL ISSUE #3: BLOCKING TRANSCRIPTION CALLS (No Timeout)

**Files**: All master scripts

**Problem**: `pipe()` or `model.transcribe()` can hang indefinitely
```python
# master_stt_fix_ov.py Line 131
result = pipe(audio_processed)  # ❌ No timeout - can hang 10-30 seconds
```

**Impact**:
- **UI freezes** during transcription
- **Audio buffer overflows** while waiting
- **Test hangs** indefinitely on CPU overload

**Fix Required**:
```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Usage:
with timeout(5):  # 5-second timeout
    try:
        result = pipe(audio_processed)
    except TimeoutError as e:
        print(f"[WARNING] Transcription timeout: {e}")
        status = "TIMEOUT"
```

---

## 🔴 CRITICAL ISSUE #4: AUDIO CLIPPING IN WAV EXPORT

**Files**: All master scripts that save audio

**Problem**: No normalization before int16 conversion
```python
# master_stt_fix.py Line 143
wf.writeframes((final_audio * 32767).astype(np.int16).tobytes())  # ❌ CLIPPING
```

**Impact**:
- **Severe distortion** if audio peaks > 1.0
- **Unintelligible playback**
- **Inconsistent transcription** between live and saved audio

**Fix Required**:
```python
# Normalize before conversion
audio_normalized = np.clip(final_audio * 0.9, -1.0, 1.0)
wf.writeframes((audio_normalized * 32767).astype(np.int16).tobytes())

# OR use RMS normalization
rms = np.sqrt(np.mean(final_audio**2))
if rms > 0:
    audio_normalized = final_audio * (0.8 / rms)
    audio_normalized = np.clip(audio_normalized, -1.0, 1.0)
```

---

## 🔴 CRITICAL ISSUE #5: SILENT EXCEPTION HANDLING

**Files**: All master scripts

**Problem**: Exceptions caught but test continues in broken state
```python
# master_stt_distil_v20.py Line 71
try:
    model = OVModelForSpeechSeq2Seq.from_pretrained(...)
except Exception as e:
    print(f"{RED}STRATEGY ERROR: {e}{RESET}")
    return  # ❌ Returns but doesn't exit - continues execution
```

**Impact**:
- **False positives**: Test appears to run but produces no results
- **Crashes later**: Code continues with uninitialized variables
- **Debugging nightmare**: Root cause hidden

**Fix Required**:
```python
except Exception as e:
    print(f"{RED}CRITICAL ERROR: {e}{RESET}")
    sys.exit(1)  # ❌ Exit immediately, don't continue
```

---

## 🔴 CRITICAL ISSUE #6: MODEL LOADED EVERY RUN (No Caching Check)

**Files**: `master_stt_policy_v18.py`, `master_stt_distil_v20.py`

**Problem**: Model quantization/loading runs every single time
```python
# master_stt_policy_v18.py Line 40
def optimize_model_policy():
    if os.path.exists(QUANT_PATH):
        return  # ✓ Has cache check
    # But still takes 2-5 minutes on first run
```

**Impact**:
- **5-10 minute delay** on first run
- **Wasted time** during testing iterations
- **Confusion** for users expecting quick tests

**Fix Required**:
```python
# Add timestamp check - re-optimize only if model is old
import time
CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days

def optimize_model_policy():
    if os.path.exists(QUANT_PATH):
        cache_time = os.path.getmtime(QUANT_PATH)
        if time.time() - cache_time < CACHE_MAX_AGE:
            print(f"{GREEN}[OK] Using cached optimized model{RESET}")
            return
    # ... optimize
```

---

## 🟡 MODERATE ISSUE #7: INCONSISTENT TEST DURATIONS

**Files**: All master scripts

| File | Duration |
|------|----------|
| `master_stt_fix.py` | 300s (5 min) |
| `master_stt_fix_ov.py` | 300s (5 min) |
| `master_stt_stateful_v19.py` | 300s (5 min) |
| `master_stt_policy_v18.py` | 300s (5 min) |
| `master_stt_distil_v20.py` | 300s (5 min) |

**Problem**: 5-minute tests are too long for quick iteration

**Fix Required**:
```python
# Add command-line override
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
args = parser.parse_args()
TEST_DURATION = args.duration
```

---

## 🟡 MODERATE ISSUE #8: NO ASSERTIONS (Not Real Tests)

**Files**: All master scripts

**Problem**: Scripts print results but never validate them
```python
print(f"{GREEN}{BOLD}FINAL SESSION SUMMARY:{RESET}")
print(f"\"{final_text}\"")  # ❌ No assertion - always "passes"
```

**Fix Required**:
```python
# Add validation
if not final_text:
    print(f"{RED}[FAIL] No transcription produced{RESET}")
    sys.exit(1)
else:
    print(f"{GREEN}[PASS] Transcription successful{RESET}")
    sys.exit(0)
```

---

## 🟡 MODERATE ISSUE #9: UNICODE NEWLINE CHARACTERS

**Files**: `master_stt_fix_ov.py`, `master_stt_stateful_v19.py`, `master_stt_policy_v18.py`, `master_stt_distil_v20.py`

**Problem**: Literal newline in string instead of escape sequence
```python
# Line 169 in master_stt_fix_ov.py
sys.stdout.write("\033[?25h
")  # ❌ Literal newline in source
print(f"
{GREEN}SESSION ENDED...{RESET}")  # ❌ Should be \n
```

**Impact**:
- **Syntax warnings** on some Python versions
- **Code style issues**
- **Potential parsing errors**

**Fix Required**:
```python
sys.stdout.write("\033[?25h\n")
print(f"\n{GREEN}SESSION ENDED...{RESET}")
```

---

## PRIORITY ORDER FOR FIXES

### Immediate (Must Fix Before Testing):
1. **Unbounded memory growth** → Use `deque` with `maxlen`
2. **Thread locks** → Add `Lock()` to all shared state
3. **Timeout on transcription** → Add 5-second timeout
4. **Audio clipping** → Normalize before int16 conversion
5. **Silent failures** → Exit on critical errors

### Secondary (Should Fix):
6. **Model caching** → Add timestamp-based cache validation
7. **Unicode newlines** → Replace literal newlines with `\n`

### Tertiary (Nice to Have):
8. **Configurable duration** → Add argparse for test length
9. **Assertions** → Add pass/fail validation

---

## FILES REQUIRING FIXES

| File | Critical Issues | Priority |
|------|-----------------|----------|
| `master_stt_fix.py` | #1, #2, #3, #4, #5 | 🔴 CRITICAL |
| `master_stt_fix_ov.py` | #1, #2, #3, #4, #5, #9 | 🔴 CRITICAL |
| `master_stt_stateful_v19.py` | #1, #2, #3, #4, #5, #9 | 🔴 CRITICAL |
| `master_stt_policy_v18.py` | #1, #2, #3, #4, #5, #6, #9 | 🔴 CRITICAL |
| `master_stt_distil_v20.py` | #1, #2, #3, #4, #5, #9 | 🔴 CRITICAL |

---

## DO NOT MODIFY

- `whisper stt/` directory (OpenAI Whisper source code)
- `KittenTTS/` directory (third-party TTS library)
- `DeepSeek-R1/` directory (documentation only)

---

## VERIFICATION AFTER FIXES

Run these commands to verify:
```bash
# Quick 30-second test (should complete without crash)
python master_stt_fix.py --duration 30

# Memory test (check for leaks)
python -m memory_profiler master_stt_fix.py

# All master scripts
for f in master_stt_*.py; do echo "Testing $f..."; python "$f" --duration 30; done
```

**Expected Results**:
- ✅ All tests complete full duration without crashes
- ✅ Memory usage stays constant (no growth)
- ✅ No thread safety errors in logs
- ✅ Clean exit with proper status codes

---

**Generated**: 2026-02-19
**For**: Gemini Agent code modifications
**Scope**: CRITICAL testing issues in master_stt_*.py files only
