#!/usr/bin/env python3
"""
confirm vllm 0.9.1 exposes metrics
"""

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ENABLE_METRICS"] = "1"
os.environ["VLLM_PROFILE"] = "1"
os.environ["VLLM_DETAILED_METRICS"] = "1"
os.environ["VLLM_REQUEST_METRICS"] = "1"

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vllm_model import VLLMModel, VLLMConfig

def test_improved_vllm_model():
    """Test the improved VLLMModel with automatic metrics setup."""
    print("🧪 Testing Improved VLLMModel with Auto-Metrics")
    print("=" * 50)
    
    config = VLLMConfig(
        model_path="microsoft/DialoGPT-small",
        gpu_memory_utilization=0.3,
        enable_metrics=True,
        vllm_version="0.9.1"
    )
    
    print(f"Creating VLLMModel with config:")
    print(f"  Model: {config.model_path}")
    print(f"  Metrics enabled: {config.enable_metrics}")
    print(f"  Version override: {config.vllm_version}")
    
    model = VLLMModel(config)
    
    info = model.get_model_info()
    print(f"\n📊 Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔮 Testing single prediction...")
    
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=16)
    completions = model.model.generate(["What is the capital of France?"], sampling_params)
    completion = completions[0]
    
    print(f"Debug - Completion attributes: {[attr for attr in dir(completion) if not attr.startswith('_')]}")
    print(f"Debug - Completion.metrics: {getattr(completion, 'metrics', None)}")
    print(f"Debug - Has request_id: {hasattr(completion, 'request_id')}")
    if hasattr(completion, 'request_id'):
        print(f"Debug - Request ID: {completion.request_id}")
    
    engine = getattr(model.model, "llm_engine", None) or getattr(model.model, "engine", None)
    print(f"Debug - Engine type: {type(engine)}")
    if engine:
        request_metrics = getattr(engine, "request_metrics", None)
        print(f"Debug - Engine.request_metrics: {type(request_metrics)}")
        if request_metrics:
            print(f"Debug - Request metrics keys: {list(request_metrics.keys()) if hasattr(request_metrics, 'keys') else 'Not a dict'}")
    
    print("\n🔮 Now testing through our wrapper...")
    result = model.predict(
        prompt="What is the capital of France?",
        max_tokens=16,
        temperature=0.7
    )
    
    print(f"✅ Prediction successful!")
    print(f"  Generated: '{result.generated_text}'")
    print(f"  Input tokens: {result.input_tokens}")
    print(f"  Output tokens: {result.output_tokens}")
    print(f"  TTFT: {result.ttft:.1f}ms")
    print(f"  Total time: {result.total_time_ms:.1f}ms")
    print(f"  Tokens/sec: {result.tokens_per_second:.1f}")
    print(f"  Metrics available: {result.metrics is not None}")
    
    if result.metrics:
        print(f"  Raw metrics type: {type(result.metrics)}")
    
    print(f"\n🎉 Test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_improved_vllm_model()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)