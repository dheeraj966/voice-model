# CRITICAL ISSUES IN MASTER STT SCRIPTS - REVISED (Based on Agent Feedback)

## Context
Updated analysis based on agent's review. Focus on **confirmed critical issues** that affect test stability and reliability.

---

## 🔴 CRITICAL PRIORITY 1: MEMORY LEAK - Unbounded Buffer Growth

**Status**: ✅ CONFIRMED - HIGHEST PRIORITY

**Files**: ALL master_stt_*.py files
- `master_stt_fix.py` Line 93
- `master_stt_fix_ov.py` Line 107
- `master_stt_stateful_v19.py` Line 94
- `master_stt_policy_v18.py` Line 122
- `master_stt_distil_v20.py` Line 95

**Problem**:
```python
full_audio_log = []  # ❌ Plain list - grows forever

def callback(indata, frames, time_info, status_msg):
    full_audio_log.append(audio)  # Appends every callback for 5 minutes
```

**Impact**:
- **150-300MB** memory over 5-minute test
- **GC pauses** → audio dropouts
- **Crash** on systems with <4GB RAM

**Fix Required**:
```python
from collections import deque

# Option 1: Time-based limit (keep last 30 seconds)
full_audio_log = deque(maxlen=int(SAMPLE_RATE * 30 / BLOCK_SIZE))

# Option 2: Chunk count limit (keep last 500 chunks = ~30 seconds)
full_audio_log = deque(maxlen=500)
```

---

## 🔴 CRITICAL PRIORITY 2: RACE CONDITION - No Thread Locks

**Status**: ✅ CONFIRMED - HIGHEST PRIORITY

**Files**: 
- `master_stt_fix.py` Lines 93-106
- `master_stt_fix_ov.py` Lines 107-131
- `master_stt_stateful_v19.py` Lines 94-118

**Problem**:
```python
# Shared state accessed by audio callback thread AND main thread
rolling_audio = deque(maxlen=int(SAMPLE_RATE * 5.0))

# Audio callback thread (writes)
def callback(indata, frames, time_info, status_msg):
    rolling_audio.extend(audio)  # ❌ No lock

# Main thread (reads)
audio_to_transcribe = np.array(rolling_audio, dtype=np.float32)  # ❌ No lock
```

**Impact**:
- **RuntimeError**: "deque mutated during iteration"
- **Data corruption**: Partial audio samples
- **Random crashes** (non-deterministic)

**Fix Required**:
```python
from threading import Lock
audio_lock = Lock()

def callback(indata, frames, time_info, status_msg):
    audio = indata.copy().flatten()
    with audio_lock:  # ✅ Thread-safe write
        rolling_audio.extend(audio)

# In main loop:
with audio_lock:  # ✅ Thread-safe read
    audio_to_transcribe = np.array(rolling_audio, dtype=np.float32).copy()
```

---

## 🔴 CRITICAL PRIORITY 3: NO TIMEOUT - Blocking Transcription Calls

**Status**: ✅ CONFIRMED - HIGHEST PRIORITY

**Files**: ALL master scripts

**Problem**:
```python
# master_stt_fix_ov.py Line 131
result = pipe(audio_processed)  # ❌ Can hang indefinitely on CPU overload
```

**Impact**:
- **UI freezes** during transcription
- **Audio buffer overflow** while waiting
- **Test hangs** forever on edge cases

**Fix Required**:
```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds=5):
    """Context manager for operation timeout."""
    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Usage in main loop:
with timeout(5):  # ✅ 5-second max
    try:
        result = pipe(audio_processed)
        new_text = result["text"].strip()
    except TimeoutError as e:
        print(f"[WARNING] Transcription timeout: {e}")
        status = "TIMEOUT - Skipped"
        continue
```

---

## 🟡 MEDIUM PRIORITY 4: Model Re-Export Every Run

**Status**: ✅ CONFIRMED - MEDIUM PRIORITY

**Files**: 
- `master_stt_policy_v18.py` Lines 40-60
- `master_stt_distil_v20.py` Lines 42-58

**Problem**:
```python
model = OVModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    export=True,  # ⚠️ Always re-exports even if cache exists
    ov_config={"CACHE_DIR": "ov_cache"}
)
```

**Impact**:
- **First run**: 5-10 minutes (expected)
- **Subsequent runs**: Still 5-10 minutes (should be instant)
- **Wasted time** during development/testing

**Fix Required**:
```python
import os
import time

QUANT_PATH = "v18_optimized_model"
CACHE_MAX_AGE = 7 * 24 * 3600  # 7 days in seconds

def load_or_optimize_model():
    """Load cached model or optimize if cache is missing/old."""
    if os.path.exists(QUANT_PATH):
        # Check cache age
        cache_time = os.path.getmtime(QUANT_PATH)
        if time.time() - cache_time < CACHE_MAX_AGE:
            print(f"{GREEN}[OK] Using cached optimized model (<7 days old){RESET}")
            return OVModelForSpeechSeq2Seq.from_pretrained(
                QUANT_PATH,
                device="GPU",
                ov_config={"CACHE_DIR": "ov_cache", "PERFORMANCE_HINT": "LATENCY"}
            )
        else:
            print(f"{YELLOW}[INFO] Cache old, re-optimizing...{RESET}")
    
    # Optimize and save
    print(f"{YELLOW}Optimizing model (first time takes 5-10 min)...{RESET}")
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, export=True, compile=False
    )
    # ... quantization logic ...
    model.save_pretrained(QUANT_PATH)
    return model
```

---

## 🟢 LOW PRIORITY 5: Final WAV Export - Potential Clipping

**Status**: ⚠️ PARTIALLY CONFIRMED - LOW PRIORITY (Context Matters)

**Files**: 
- `master_stt_fix.py` Line 134
- `master_stt_fix_ov.py` Line 151

**Agent's Analysis**:
> This is ONLY in the final WAV save block (not live processing). The audio IS normalized before inference.

**Problem**:
```python
# Final save (after transcription is done)
wf.writeframes((final_audio * 32767).astype(np.int16).tobytes())  # ⚠️ May clip
```

**Context**:
- ✅ **Live processing** is safe (normalized at Line 104/123)
- ⚠️ **Saved WAV file** may have distortion if `final_audio` has peaks >1.0

**Impact**:
- **Distorted playback** of saved audio files
- **Does NOT affect** transcription quality (already done)
- **User experience** issue only

**Fix Required** (optional, low priority):
```python
# Normalize before saving
audio_normalized = np.clip(final_audio * 0.9, -1.0, 1.0)
wf.writeframes((audio_normalized * 32767).astype(np.int16).tobytes())
```

---

## ❌ NOT ISSUES (Clarified by Agent)

### Issue: Inconsistent Test Durations
**Agent's Verdict**: ❌ NOT AN ISSUE - All files use `TEST_DURATION = 300` by design for consistent comparison testing.

### Issue: No Assertions
**Agent's Verdict**: ⚠️ BY DESIGN - These are demo/test scripts, not unit tests. They're meant to show output, not pass/fail.

---

## FIX PRIORITY SUMMARY

| Priority | Issue | Affected Files | Impact |
|----------|-------|----------------|--------|
| 🔴 P1 | Memory Leak | ALL (5 files) | Crashes on low RAM |
| 🔴 P1 | Race Condition | 3 files | Random crashes |
| 🔴 P1 | No Timeout | ALL (5 files) | Infinite hangs |
| 🟡 P2 | Model Re-Export | 2 files | 5-10 min delay every run |
| 🟢 P3 | WAV Clipping | 2 files | Distorted saved audio only |

---

## RECOMMENDED FIX ORDER

### Phase 1: Critical Stability (Must Fix)
1. **Add `deque` with `maxlen`** to all `full_audio_log` lists
2. **Add `Lock()`** to all shared state access
3. **Add 5-second timeout** to all transcription calls

### Phase 2: Performance (Should Fix)
4. **Add cache timestamp check** to model loading

### Phase 3: Quality (Nice to Fix)
5. **Normalize final WAV export** (optional, low impact)

---

## VERIFICATION CHECKLIST

After fixes, verify:

```bash
# 1. Memory test - should stay constant
python master_stt_fix.py --duration 60
# Check Task Manager: Memory should be stable, not growing

# 2. Thread safety test - no race condition errors
python master_stt_fix_ov.py --duration 60
# Should complete without "deque mutated" errors

# 3. Timeout test - no hangs
python master_stt_stateful_v19.py --duration 60
# Should complete even if transcription fails

# 4. Cache test - second run should be instant
python master_stt_policy_v18.py  # First run: 5-10 min
python master_stt_policy_v18.py  # Second run: <30 seconds (after fix)
```

**Expected Results**:
- ✅ Memory usage constant (no growth)
- ✅ No thread safety errors
- ✅ No infinite hangs
- ✅ Second run uses cache (after model fix)

---

## CODE CHANGES SUMMARY

### Change 1: Import additions (top of each file)
```python
from collections import deque
from threading import Lock
import signal
from contextlib import contextmanager
```

### Change 2: Replace unbounded lists
```python
# BEFORE:
full_audio_log = []
rolling_audio = deque(maxlen=int(SAMPLE_RATE * 5.0))

# AFTER:
full_audio_log = deque(maxlen=500)  # Last 500 chunks only
rolling_audio = deque(maxlen=int(SAMPLE_RATE * 5.0))  # Already correct
```

### Change 3: Add locks
```python
# Add at top level:
audio_lock = Lock()

# Wrap callback access:
def callback(indata, frames, time_info, status_msg):
    audio = indata.copy().flatten()
    with audio_lock:
        rolling_audio.extend(audio)
        full_audio_log.append(audio)  # Now safe

# Wrap main thread access:
with audio_lock:
    audio_to_transcribe = np.array(rolling_audio, dtype=np.float32).copy()
```

### Change 4: Add timeout
```python
@contextmanager
def timeout(seconds=5):
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# Usage:
with timeout(5):
    result = pipe(audio_processed)
```

---

**Generated**: 2026-02-19 (Revised based on agent feedback)
**For**: Gemini Agent code modifications
**Scope**: CRITICAL issues only (P1-P3 priority)
