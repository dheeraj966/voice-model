# URGENT FIX: beam_size Parameter Error in master_stt_fix_ov.py

## Issue Status: ✅ CONFIRMED - CRITICAL BLOCKER

**File**: `master_stt_fix_ov.py`

**Error Message**:
```
OPEN-VINO LOAD ERROR: The following `model_kwargs` are not used by the model: ['beam_size'] 
(note: typos in the generate arguments will also show up in this list)
```

**Additional Warnings**:
```
Device set to use cpu
`return_token_timestamps` is deprecated for WhisperFeatureExtractor and will be removed in Transformers v5
`forced_decoder_ids` is deprecated in favor of the `task` and `language` flags/config options
`generation_config` default values have been modified to match model-specific defaults
```

---

## Root Cause

The `pipeline()` call is missing the `generate_kwargs` parameter that was previously present:

**Previous Code (WORKING)**:
```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device="cpu",
    generate_kwargs={"beam_size": BEAM_SIZE, "best_of": BEAM_SIZE}  # ✅ Was here
)
```

**Current Code (BROKEN)**:
```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device="cpu"  # ❌ generate_kwargs missing!
)
```

---

## Fix Required

### Option 1: Add generate_kwargs to pipeline (RECOMMENDED)

```python
# Line 88-95 - Fix the pipeline initialization
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device="cpu",
    generate_kwargs={
        "beam_size": BEAM_SIZE,      # ✅ Restore this
        "best_of": BEAM_SIZE,        # ✅ Restore this
        "language": "en",            # ✅ Use new language flag (not forced_decoder_ids)
        "task": "transcribe"         # ✅ Use new task flag
    }
)
```

### Option 2: Pass generate_kwargs during inference call

```python
# Line 131 - Pass generate_kwargs at inference time
result = pipe(
    audio_processed,
    generate_kwargs={
        "beam_size": BEAM_SIZE,
        "best_of": BEAM_SIZE,
        "language": "en",
        "task": "transcribe"
    }
)
```

---

## Additional Deprecation Warnings to Fix

### Warning 1: `return_token_timestamps` deprecated
**Status**: ⚠️ Will be removed in Transformers v5

**Current**: Not explicitly used (may be in pipeline defaults)

**Fix**: No action needed yet, but be aware for future updates

### Warning 2: `forced_decoder_ids` deprecated
**Status**: ⚠️ Deprecated in favor of `language` and `language` flags

**Current**: May be set in model's generation_config

**Fix**: Explicitly set in `generate_kwargs`:
```python
generate_kwargs={
    "language": "en",    # ✅ New way
    "task": "transcribe", # ✅ New way
    # Remove any forced_decoder_ids usage
}
```

### Warning 3: `suppress_tokens` and `begin_suppress_tokens` defaults changed
**Status**: ℹ️ Informational - model auto-set these

**Fix**: Can explicitly set if needed:
```python
generate_kwargs={
    "suppress_tokens": [1, 2, 7, 8, 9, 10, 14, 25, ...],  # Use model defaults
    "begin_suppress_tokens": [220, 50256]
}
```

---

## Complete Fixed Code Section

Replace lines 88-102 with:

```python
# Create the pipeline with beam search for nuance capture
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device="cpu",  # Pipeline runs on CPU, model is on GPU
    generate_kwargs={
        "beam_size": BEAM_SIZE,      # ✅ Restore: Maximum search depth for nuance
        "best_of": BEAM_SIZE,        # ✅ Restore: Number of candidates
        "language": "en",            # ✅ New: Explicit language (replaces forced_decoder_ids)
        "task": "transcribe",        # ✅ New: Explicit task type
        "length_penalty": 1.0,       # Optional: Length normalization
        "return_timestamps": False   # Optional: Disable timestamps for speed
    }
)

print(f"{CYAN}[3/4] Pre-Warming GPU with dummy data...{RESET}")
# Run a tiny dummy buffer to initialize the execution units
pipe(np.zeros(1600, dtype=np.float32))
```

---

## Why This Happened

The code was likely modified to remove `generate_kwargs` during debugging, but this parameter is **required** for:
1. **Beam search configuration** (`beam_size`, `best_of`)
2. **Language/task specification** (`language`, `task`)
3. **Generation constraints** (`suppress_tokens`, etc.)

Without it, the pipeline uses greedy decoding (beam_size=1) and may not capture nuances.

---

## Verification After Fix

Run the script and check for:

```bash
python master_stt_fix_ov.py
```

**Expected Output** (SUCCESS):
```
Waking up Intel Iris Xe GPU Engine (openai/whisper-medium.en)...
[1/4] Loading and Compiling Model for GPU...
      (Note: First time takes 5-10 mins. Subsequent runs are fast.)
[2/4] GPU Compilation Successful.
[3/4] Pre-Warming GPU with dummy data...
[4/4] AI Ready (Iris Xe - MEDIUM MODEL - Ultra Sensitive).
Opening Microphone Stream...
```

**No errors** about `beam_size` or unused model_kwargs.

**Warnings that are OK to ignore** (for now):
- `return_token_timestamps` deprecated (v5 removal)
- `generation_config` defaults modified (informational)

---

## Impact if Not Fixed

- ❌ Script **crashes immediately** on startup
- ❌ Cannot test any other functionality
- ❌ GPU compilation wasted (5-10 minutes lost)
- ❌ **BLOCKS ALL TESTING**

---

## Priority: 🔴 CRITICAL - BLOCKS EXECUTION

This is the **highest priority** fix because:
1. Prevents script from running at all
2. Wastes 5-10 minutes of GPU compilation time
3. Blocks testing of all other fixes

**Fix Time**: < 2 minutes (add 4 lines)
**Impact**: Enables all testing to proceed

---

**Generated**: 2026-02-19
**For**: Gemini Agent - URGENT FIX
**File**: `master_stt_fix_ov.py` lines 88-95
