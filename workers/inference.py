import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# NVML for real driver-level GPU stats (utilization %, true VRAM)
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception as e:
    print(f"[inference] NVML unavailable: {e}")
    _NVML_OK = False

_tokenizer = None
_model = None
_device = None
_device_idx = None  # int form of cuda index, e.g. 0


def load_model(model_path: str, device: str = "cuda:0"):
    """Load the tokenizer and model onto the given device."""
    global _tokenizer, _model, _device, _device_idx

    _device = device if torch.cuda.is_available() else "cpu"
    print(f"[inference] Loading {model_path} on {_device}")

    if "cuda" in _device:
        _device_idx = int(_device.split(":")[1])
    else:
        _device_idx = None

    _tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Required for batched generation: pad token + left padding for causal LMs
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if "cuda" in _device else torch.float32,
    ).to(_device)
    _model.eval()

    if "cuda" in _device:
        torch.cuda.synchronize(_device)
        resident_mb = torch.cuda.memory_allocated(_device) / 1024**2
        print(f"[inference] Model loaded. Resident VRAM: {resident_mb:.1f} MB")


def _build_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely in English."},
        {"role": "user", "content": question},
    ]
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_response(question: str, max_new_tokens: int = 256) -> dict:
    """
    Generate a response to a single question.
    Returns a dict with the response text and per-request GPU stats.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Reset PyTorch's peak memory tracker so we can measure THIS request only
    if "cuda" in _device:
        torch.cuda.reset_peak_memory_stats(_device)
        mem_before = torch.cuda.memory_allocated(_device)

    t0 = time.time()

    prompt = _build_prompt(question)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    input_tokens = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    if "cuda" in _device:
        torch.cuda.synchronize(_device)

    new_tokens = output_ids[0][input_tokens:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    output_tokens = new_tokens.shape[-1]

    latency_ms = (time.time() - t0) * 1000
    tps = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

    stats = {
        "latency_ms": round(latency_ms, 1),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "tokens_per_sec": round(tps, 1),
    }

    if "cuda" in _device:
        peak = torch.cuda.max_memory_allocated(_device)
        stats["peak_mem_mb"] = round(peak / 1024**2, 1)
        stats["request_mem_mb"] = round((peak - mem_before) / 1024**2, 1)

    return {"response": text, "stats": stats}


def generate_batch(questions: list, max_new_tokens: int = 256) -> list:
    """
    Generate responses for a batch of questions in a single GPU call.
    Returns a list of dicts, one per input question, in the same order.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    if not questions:
        return []

    if "cuda" in _device:
        torch.cuda.reset_peak_memory_stats(_device)
        mem_before = torch.cuda.memory_allocated(_device)

    t0 = time.time()

    prompts = [_build_prompt(q) for q in questions]
    inputs = _tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(_device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    if "cuda" in _device:
        torch.cuda.synchronize(_device)

    latency_ms = (time.time() - t0) * 1000

    results = []
    total_out_tokens = 0
    for i in range(output_ids.shape[0]):
        new_tokens = output_ids[i][input_len:]
        out_count = (new_tokens != _tokenizer.pad_token_id).sum().item()
        total_out_tokens += out_count
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        results.append({
            "response": text,
            "stats": {
                "latency_ms": round(latency_ms, 1),  # whole-batch latency
                "output_tokens": int(out_count),
                "batch_size": len(questions),
            }
        })

    # Tag the batch-level memory cost on the first item (you can also broadcast)
    if "cuda" in _device and results:
        peak = torch.cuda.max_memory_allocated(_device)
        batch_peak_mb = round(peak / 1024**2, 1)
        batch_added_mb = round((peak - mem_before) / 1024**2, 1)
        per_request_mb = round(batch_added_mb / len(questions), 1)
        for r in results:
            r["stats"]["peak_mem_mb"] = batch_peak_mb
            r["stats"]["batch_added_mem_mb"] = batch_added_mb
            r["stats"]["per_request_mem_mb"] = per_request_mb
        results[0]["stats"]["batch_total_output_tokens"] = total_out_tokens
        results[0]["stats"]["batch_tokens_per_sec"] = round(
            total_out_tokens / (latency_ms / 1000), 1
        )

    return results


def get_gpu_stats() -> dict:
    """
    Return current GPU stats. Combines PyTorch's tracked memory with NVML's
    real driver-level utilization and VRAM info.
    """
    if not _device or "cuda" not in _device:
        return {"device": "cpu"}

    stats = {
        "device": _device,
        "gpu_name": torch.cuda.get_device_name(_device_idx),
        # PyTorch process-level memory
        "torch_mem_allocated_mb": round(
            torch.cuda.memory_allocated(_device_idx) / 1024**2, 1
        ),
        "torch_mem_reserved_mb": round(
            torch.cuda.memory_reserved(_device_idx) / 1024**2, 1
        ),
    }

    # Driver-level: utilization %, total VRAM, true used VRAM (all processes)
    if _NVML_OK:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(_device_idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats.update({
                "gpu_util_pct": util.gpu,
                "gpu_mem_bus_util_pct": util.memory,
                "vram_used_mb": round(mem.used / 1024**2, 1),
                "vram_total_mb": round(mem.total / 1024**2, 1),
                "vram_free_mb": round(mem.free / 1024**2, 1),
                "gpu_temp_c": temp,
            })
        except Exception as e:
            stats["nvml_error"] = str(e)

    return stats