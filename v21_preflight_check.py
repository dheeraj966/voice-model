"""
V21 PREFLIGHT CHECK - PERFORMANCE BENCHMARK
Tests the feasibility of Beam Search (Nuance) vs. Latency on Intel Iris Xe.
"""

import time
import torch
import numpy as np
from transformers import AutoProcessor, pipeline
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
import webrtcvad

# Constants
MODEL_ID = "distil-whisper/distil-medium.en"
ASSISTANT_MODEL_ID = "openai/whisper-tiny.en"
SAMPLE_RATE = 16000

def benchmark_inference(pipe, audio_input, beam_size):
    start = time.time()
    # Explicitly pass beam_size through generate_kwargs
    result = pipe(audio_input, generate_kwargs={"beam_size": beam_size, "assistant_model": pipe.model.assistant_model if hasattr(pipe.model, 'assistant_model') else None})
    end = time.time()
    return end - start, result["text"]

def main():
    print("--- V21 PREFLIGHT: HARDWARE VALIDATION ---")
    
    # 1. Test VAD Logic
    print("\n[1/3] Testing WebRTC VAD Compatibility...")
    try:
        vad = webrtcvad.Vad(3)
        dummy_audio = np.zeros(480, dtype=np.int16).tobytes() # 30ms at 16kHz
        is_speech = vad.is_speech(dummy_audio, SAMPLE_RATE)
        print(f"      VAD initialized and tested: OK (Silence detected as {is_speech})")
    except Exception as e:
        print(f"      VAD Test Failed: {e}")

    # 2. Load Models for Benchmarking
    print("\n[2/3] Loading Models to GPU (Iris Xe)...")
    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, device="GPU", ov_config={"PERFORMANCE_HINT": "LATENCY"}, export=True
        )
        assistant = OVModelForSpeechSeq2Seq.from_pretrained(
            ASSISTANT_MODEL_ID, device="CPU", export=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        
        # Configure pipeline with assistant
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            generate_kwargs={"assistant_model": assistant}
        )
        # Manually attach for benchmark function if needed
        pipe.model.assistant_model = assistant
        
        print("      Models Loaded: OK")
    except Exception as e:
        print(f"      Model Load Failed: {e}")
        return

    # 3. Benchmark Beam Sizes
    print("\n[3/3] Benchmarking Latency vs. Beam Size...")
    # Create 1 second of dummy "speech"
    dummy_input = np.random.uniform(-1, 1, SAMPLE_RATE).astype(np.float32)
    
    for b in [1, 2]:
        latencies = []
        for i in range(3): # Warmup and average
            lat, _ = benchmark_inference(pipe, dummy_input, b)
            if i > 0: # Skip first warmup run
                latencies.append(lat)
        avg_lat = sum(latencies) / len(latencies)
        print(f"      Beam Size {b}: Avg Latency = {avg_lat:.3f}s")
        
    print("\n--- PREFLIGHT COMPLETE ---")
    print("If Beam Size 2 Avg Latency < 0.600s, it is safe for V21 Nuance Master.")

if __name__ == "__main__":
    main()
