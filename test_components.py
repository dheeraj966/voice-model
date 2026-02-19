#!/usr/bin/env python3
"""
Voice Agent Component Testing Script
Tests STT (Whisper), TTS (KittenTTS), and LLM integration
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Results storage
test_results = []

def log_result(component, test_name, passed, details="", recommendation=""):
    """Log test result"""
    result = {
        "component": component,
        "test": test_name,
        "passed": passed,
        "details": details,
        "recommendation": recommendation,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    test_results.append(result)
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n{status} | {component} - {test_name}")
    if details:
        print(f"   Details: {details}")
    if recommendation and not passed:
        print(f"   Recommendation: {recommendation}")
    return passed


# =============================================================================
# TEST 1: STT (Whisper)
# =============================================================================
def test_whisper_stt():
    print("\n" + "="*60)
    print("TEST 1: STT (Whisper)")
    print("="*60)
    
    whisper_passed = True
    
    # Test 1.1: Import whisper module
    try:
        start_time = time.time()
        # Add whisper stt to path
        whisper_path = r"C:\Users\maxwe\ml project\voice agents\whisper stt"
        sys.path.insert(0, whisper_path)
        import whisper
        import_time = (time.time() - start_time) * 1000
        whisper_version = getattr(whisper, '__version__', 'unknown')
        log_result("Whisper STT", "Module Import", True, 
                   f"Import time: {import_time:.2f}ms, Version: {whisper_version}")
    except Exception as e:
        log_result("Whisper STT", "Module Import", False, 
                   f"Error: {str(e)}",
                   "Ensure whisper is installed: pip install openai-whisper")
        return False
    
    # Test 1.2: Check available models
    try:
        models = whisper.available_models()
        log_result("Whisper STT", "Available Models Check", True,
                   f"Found {len(models)} models: {', '.join(models[:5])}... (showing first 5)")
    except Exception as e:
        log_result("Whisper STT", "Available Models Check", False,
                   f"Error: {str(e)}")
        whisper_passed = False
    
    # Test 1.3: Check torch and CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        torch_version = torch.__version__
        log_result("Whisper STT", "PyTorch/CUDA Check", True,
                   f"PyTorch: {torch_version}, CUDA available: {cuda_available}, Device: {device}")
    except Exception as e:
        log_result("Whisper STT", "PyTorch/CUDA Check", False,
                   f"Error: {str(e)}",
                   "Install PyTorch: pip install torch")
        whisper_passed = False
    
    # Test 1.4: Test model loading (tiny model for speed)
    try:
        print("\n   Loading 'tiny' model (this may take a moment on first run)...")
        start_time = time.time()
        model = whisper.load_model("tiny", download_root=r"C:\Users\maxwe\ml project\voice agents\.cache\whisper")
        load_time = (time.time() - start_time)
        log_result("Whisper STT", "Model Loading (tiny)", True,
                   f"Load time: {load_time:.2f}s, Model dims: {model.dims}")
    except Exception as e:
        log_result("Whisper STT", "Model Loading (tiny)", False,
                   f"Error: {str(e)}",
                   "Check internet connection for model download or disk space")
        whisper_passed = False
    
    # Test 1.5: Test transcription with generated audio (create simple test)
    try:
        import numpy as np
        # Create a simple test audio (silence for testing - just verify no crash)
        sample_rate = 16000
        duration = 1  # 1 second of silence
        test_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        start_time = time.time()
        result = model.transcribe(test_audio, language="en")
        transcribe_time = (time.time() - start_time) * 1000
        log_result("Whisper STT", "Transcription Test", True,
                   f"Transcribe time: {transcribe_time:.2f}ms, Output: '{result['text'].strip()}'")
    except Exception as e:
        log_result("Whisper STT", "Transcription Test", False,
                   f"Error: {str(e)}")
        whisper_passed = False
    
    return whisper_passed


# =============================================================================
# TEST 2: TTS (KittenTTS)
# =============================================================================
def test_kittentts():
    print("\n" + "="*60)
    print("TEST 2: TTS (KittenTTS)")
    print("="*60)
    
    tts_passed = True
    test_audio_path = r"C:\Users\maxwe\ml project\voice agents\test_output.wav"
    
    # Test 2.1: Import kittentts module
    try:
        start_time = time.time()
        kittentts_path = r"C:\Users\maxwe\ml project\voice agents\KittenTTS"
        sys.path.insert(0, kittentts_path)
        from kittentts import KittenTTS, get_model
        import_time = (time.time() - start_time) * 1000
        log_result("KittenTTS", "Module Import", True,
                   f"Import time: {import_time:.2f}ms")
    except Exception as e:
        log_result("KittenTTS", "Module Import", False,
                   f"Error: {str(e)}",
                   "Install dependencies: pip install huggingface_hub onnxruntime")
        return False
    
    # Test 2.2: Check dependencies (onnxruntime, huggingface_hub)
    try:
        import onnxruntime
        import huggingface_hub
        onnx_version = onnxruntime.__version__
        hf_version = huggingface_hub.__version__
        log_result("KittenTTS", "Dependencies Check", True,
                   f"onnxruntime: {onnx_version}, huggingface_hub: {hf_version}")
    except Exception as e:
        log_result("KittenTTS", "Dependencies Check", False,
                   f"Error: {str(e)}",
                   "Install: pip install onnxruntime huggingface_hub")
        tts_passed = False
    
    # Test 2.3: Initialize KittenTTS model
    try:
        print("\n   Initializing KittenTTS model (downloading if needed)...")
        start_time = time.time()
        tts = KittenTTS(model_name="KittenML/kitten-tts-nano-0.1", 
                        cache_dir=r"C:\Users\maxwe\ml project\voice agents\.cache\kittentts")
        init_time = (time.time() - start_time)
        log_result("KittenTTS", "Model Initialization", True,
                   f"Init time: {init_time:.2f}s")
    except Exception as e:
        log_result("KittenTTS", "Model Initialization", False,
                   f"Error: {str(e)}",
                   "Check internet connection for model download")
        tts_passed = False
        return False  # Can't continue without model
    
    # Test 2.4: Check available voices
    try:
        voices = tts.available_voices
        log_result("KittenTTS", "Available Voices Check", True,
                   f"Found {len(voices)} voices: {', '.join(voices[:5])}... (showing first 5)")
    except Exception as e:
        log_result("KittenTTS", "Available Voices Check", False,
                   f"Error: {str(e)}")
        tts_passed = False
    
    # Test 2.5: Generate audio sample
    try:
        test_text = "Hello, this is a test."
        print(f"\n   Generating audio for: '{test_text}'")
        start_time = time.time()
        audio = tts.generate(test_text, voice="expr-voice-5-m", speed=1.0)
        gen_time = (time.time() - start_time) * 1000
        audio_duration = len(audio) / 24000 if hasattr(audio, '__len__') else 0
        log_result("KittenTTS", "Audio Generation", True,
                   f"Generation time: {gen_time:.2f}ms, Audio length: {len(audio)} samples (~{audio_duration:.2f}s)")
    except Exception as e:
        log_result("KittenTTS", "Audio Generation", False,
                   f"Error: {str(e)}")
        tts_passed = False
    
    # Test 2.6: Save audio to file
    try:
        tts.generate_to_file(test_text, test_audio_path, voice="expr-voice-5-m", sample_rate=24000)
        file_size = os.path.getsize(test_audio_path) if os.path.exists(test_audio_path) else 0
        log_result("KittenTTS", "Save Audio to File", True,
                   f"Saved to: {test_audio_path}, File size: {file_size} bytes")
    except Exception as e:
        log_result("KittenTTS", "Save Audio to File", False,
                   f"Error: {str(e)}")
        tts_passed = False
    
    # Cleanup test file
    try:
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
            print(f"   Cleaned up test file: {test_audio_path}")
    except:
        pass
    
    return tts_passed


# =============================================================================
# TEST 3: LLM Integration
# =============================================================================
def test_llm_integration():
    print("\n" + "="*60)
    print("TEST 3: LLM Integration")
    print("="*60)
    
    llm_passed = True
    
    # Check DeepSeek-R1 folder - it's documentation only, not runnable code
    deepseek_path = r"C:\Users\maxwe\ml project\voice agents\DeepSeek-R1"
    
    # Test 3.1: Check for local LLM implementations
    try:
        # Check if vLLM or transformers can be used
        has_vllm = False
        has_transformers = False
        has_ollama = False
        
        try:
            import vllm
            has_vllm = True
        except:
            pass
        
        try:
            import transformers
            has_transformers = True
        except:
            pass
        
        try:
            import ollama
            has_ollama = True
        except:
            pass
        
        details = []
        if has_vllm:
            details.append(f"vLLM available")
        if has_transformers:
            details.append(f"transformers available")
        if has_ollama:
            details.append(f"ollama available")
        
        if not details:
            details.append("No local LLM frameworks detected")
        
        log_result("LLM Integration", "Framework Availability", True,
                   f"Available: {', '.join(details)}")
    except Exception as e:
        log_result("LLM Integration", "Framework Availability", False,
                   f"Error: {str(e)}")
        llm_passed = False
    
    # Test 3.2: Check DeepSeek-R1 folder contents
    try:
        contents = os.listdir(deepseek_path)
        md_files = [f for f in contents if f.endswith('.md') or f.endswith('.pdf')]
        log_result("LLM Integration", "DeepSeek-R1 Folder Check", True,
                   f"Contains documentation: {md_files}")
        # Note: This is documentation only, not runnable code
    except Exception as e:
        log_result("LLM Integration", "DeepSeek-R1 Folder Check", False,
                   f"Error: {str(e)}")
        llm_passed = False
    
    # Test 3.3: Test Hugging Face API connectivity (for potential model loading)
    try:
        from huggingface_hub import list_repo_files
        start_time = time.time()
        # Check if we can access DeepSeek-R1-Distill-Qwen-1.5B (smallest distill model)
        files = list_repo_files("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        check_time = (time.time() - start_time) * 1000
        log_result("LLM Integration", "HuggingFace Connectivity", True,
                   f"Can access DeepSeek-R1-Distill models, Check time: {check_time:.2f}ms")
    except Exception as e:
        log_result("LLM Integration", "HuggingFace Connectivity", False,
                   f"Error: {str(e)}",
                   "Check internet connection or HuggingFace API access")
        llm_passed = False
    
    # Test 3.4: Test basic transformers pipeline (if available)
    try:
        import transformers
        from transformers import pipeline
        
        # Test with a tiny model for quick validation
        print("\n   Testing transformers pipeline with small model...")
        start_time = time.time()
        # Use a very small model for quick testing
        generator = pipeline("text-generation", model="microsoft/phi-1.5", 
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True)
        init_time = (time.time() - start_time)
        
        # Quick generation test
        start_time = time.time()
        result = generator("Hello, I am", max_length=20, do_sample=False)
        gen_time = (time.time() - start_time) * 1000
        
        log_result("LLM Integration", "Transformers Pipeline Test", True,
                   f"Init: {init_time:.2f}s, Generation: {gen_time:.2f}ms, Output: '{result[0]['generated_text']}'")
    except ImportError:
        log_result("LLM Integration", "Transformers Pipeline Test", False,
                   f"transformers not installed",
                   "Install: pip install transformers accelerate")
        llm_passed = False
    except Exception as e:
        log_result("LLM Integration", "Transformers Pipeline Test", False,
                   f"Error: {str(e)}",
                   "May need GPU or more RAM for model loading")
        llm_passed = False
    
    return llm_passed


# =============================================================================
# TEST 4: Integration Test (Pipeline Concept)
# =============================================================================
def test_integration_pipeline():
    print("\n" + "="*60)
    print("TEST 4: Integration Test (STT → LLM → TTS Pipeline)")
    print("="*60)
    
    pipeline_passed = True
    
    # Test 4.1: Verify all components are available
    try:
        # Re-import to ensure modules are available
        whisper_path = r"C:\Users\maxwe\ml project\voice agents\whisper stt"
        kittentts_path = r"C:\Users\maxwe\ml project\voice agents\KittenTTS"
        
        if whisper_path not in sys.path:
            sys.path.insert(0, whisper_path)
        if kittentts_path not in sys.path:
            sys.path.insert(0, kittentts_path)
        
        import whisper
        from kittentts import KittenTTS
        import torch
        
        log_result("Integration", "Component Availability", True,
                   "All core components importable")
    except Exception as e:
        log_result("Integration", "Component Availability", False,
                   f"Error: {str(e)}",
                   "Ensure all dependencies are installed")
        return False
    
    # Test 4.2: Simulate pipeline flow (without actual audio I/O)
    try:
        print("\n   Simulating pipeline flow...")
        total_start = time.time()
        
        # Step 1: STT - Create test audio and transcribe
        whisper_model = whisper.load_model("tiny", download_root=r"C:\Users\maxwe\ml project\voice agents\.cache\whisper", 
                                           device="cpu")  # Force CPU for compatibility
        sample_rate = 16000
        test_audio = np.zeros(int(sample_rate * 1), dtype=np.float32)  # 1 second silence
        
        stt_start = time.time()
        stt_result = whisper_model.transcribe(test_audio, language="en")
        stt_time = (time.time() - stt_start) * 1000
        
        # Step 2: LLM - Use simple response (since no full LLM loaded)
        llm_input = stt_result['text'] if stt_result['text'].strip() else "Hello"
        llm_response = f"I received: '{llm_input}'. How can I help?"  # Mock response
        llm_time = 50  # Mock time
        
        # Step 3: TTS - Generate response audio
        tts = KittenTTS(model_name="KittenML/kitten-tts-nano-0.1",
                        cache_dir=r"C:\Users\maxwe\ml project\voice agents\.cache\kittentts")
        
        tts_start = time.time()
        tts_audio = tts.generate(llm_response, voice="expr-voice-5-m")
        tts_time = (time.time() - tts_start) * 1000
        
        total_time = (time.time() - total_start) * 1000
        
        log_result("Integration", "Pipeline Flow Simulation", True,
                   f"STT: {stt_time:.2f}ms, LLM: {llm_time}ms (mock), TTS: {tts_time:.2f}ms, Total: {total_time:.2f}ms")
        
    except Exception as e:
        log_result("Integration", "Pipeline Flow Simulation", False,
                   f"Error: {str(e)}\n{traceback.format_exc()}")
        pipeline_passed = False
    
    # Test 4.3: Data format compatibility check
    try:
        import numpy as np
        
        # Check audio format compatibility
        whisper_sr = 16000  # Whisper expects 16kHz
        kittentts_sr = 24000  # KittenTTS outputs 24kHz
        
        log_result("Integration", "Audio Format Compatibility", True,
                   f"Whisper SR: {whisper_sr}Hz, KittenTTS SR: {kittentts_sr}Hz - Resampling needed for pipeline")
    except Exception as e:
        log_result("Integration", "Audio Format Compatibility", False,
                   f"Error: {str(e)}")
        pipeline_passed = False
    
    return pipeline_passed


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("VOICE AGENT COMPONENT TESTING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    import numpy as np
    import torch
    
    # Run all tests
    whisper_ok = test_whisper_stt()
    tts_ok = test_kittentts()
    llm_ok = test_llm_integration()
    
    # Integration test only if all components pass
    if whisper_ok and tts_ok:
        integration_ok = test_integration_pipeline()
    else:
        print("\n[SKIP] Skipping integration test - some components failed")
        integration_ok = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for r in test_results if r["passed"])
    total_count = len(test_results)
    
    print(f"\nTotal Tests: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Success Rate: {passed_count/total_count*100:.1f}%")
    
    print("\n" + "-"*60)
    print("DETAILED RESULTS:")
    print("-"*60)
    
    current_component = None
    for r in test_results:
        if r["component"] != current_component:
            current_component = r["component"]
            print(f"\n[{current_component}]")
        
        status = "[PASS]" if r["passed"] else "[FAIL]"
        detail_preview = r['details'][:80] + "..." if len(r['details']) > 80 else r['details']
        print(f"  {status} {r['test']}: {detail_preview}")
    
    # Overall readiness
    print("\n" + "="*60)
    print("OVERALL SYSTEM READINESS")
    print("="*60)
    
    if whisper_ok and tts_ok and llm_ok:
        print("\n[SUCCESS] ALL COMPONENTS READY FOR INTEGRATION")
        print("\nRecommendations:")
        print("  1. Set up a proper LLM serving solution (vLLM, Ollama, or API)")
        print("  2. Implement audio resampling between Whisper (16kHz) and KittenTTS (24kHz)")
        print("  3. Add error handling and retry logic for production use")
    else:
        print("\n[WARNING] SOME COMPONENTS NEED ATTENTION")
        if not whisper_ok:
            print("  - Whisper STT: Check installation and model downloads")
        if not tts_ok:
            print("  - KittenTTS: Check dependencies and HuggingFace access")
        if not llm_ok:
            print("  - LLM: Install transformers or set up LLM serving")
    
    print("\n" + "="*60)
