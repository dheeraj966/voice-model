"""
LLM Integration Test Script
Tests available LLM components.
"""

import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 50)
print("LLM INTEGRATION TEST")
print("=" * 50)

results = {}

# Test 1: Transformers (HuggingFace)
print("\n1. Testing Transformers (HuggingFace)...")
try:
    from transformers import pipeline
    print("   [OK] Transformers imported successfully!")
    
    # Test loading a small model
    print("   Loading small sentiment model...")
    classifier = pipeline("sentiment-analysis")
    result = classifier("This is a test")[0]
    print(f"   [OK] Test result: {result['label']} ({result['score']:.2f})")
    results['transformers'] = 'PASS'
except Exception as e:
    print(f"   [FAIL] {e}")
    results['transformers'] = 'FAIL'

# Test 2: vLLM (Local LLM inference)
print("\n2. Testing vLLM...")
try:
    import vllm
    print(f"   [OK] vLLM imported successfully! (v{vllm.__version__})")
    results['vllm'] = 'PASS'
except Exception as e:
    print(f"   [FAIL] {e}")
    results['vllm'] = 'FAIL'

# Test 3: LangChain (LLM orchestration)
print("\n3. Testing LangChain...")
try:
    from langchain.llms import FakeListLLM
    print("   [OK] LangChain imported successfully!")
    results['langchain'] = 'PASS'
except Exception as e:
    print(f"   [FAIL] {e}")
    results['langchain'] = 'FAIL'

# Test 4: OpenAI API (for cloud LLMs like DeepSeek)
print("\n4. Testing OpenAI API client...")
try:
    from openai import OpenAI
    print("   [OK] OpenAI client imported successfully!")
    results['openai'] = 'PASS'
except Exception as e:
    print(f"   [FAIL] {e}")
    results['openai'] = 'FAIL'

# Test 5: Accelerate (model acceleration)
print("\n5. Testing Accelerate...")
try:
    import accelerate
    print(f"   [OK] Accelerate imported successfully! (v{accelerate.__version__})")
    results['accelerate'] = 'PASS'
except Exception as e:
    print(f"   [FAIL] {e}")
    results['accelerate'] = 'FAIL'

# Summary
print("\n" + "=" * 50)
print("LLM TEST SUMMARY")
print("=" * 50)
for component, status in results.items():
    symbol = "[OK]" if status == 'PASS' else "[FAIL]"
    print(f"{symbol} {component}: {status}")

passed = sum(1 for s in results.values() if s == 'PASS')
total = len(results)
print(f"\nPassed: {passed}/{total}")
print("=" * 50)
