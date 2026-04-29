import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None
_device = None


def load_model(model_path: str):
    global _tokenizer, _model, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    ).to(_device)
    _model.eval()


def generate_response(question: str, max_new_tokens: int = 256) -> str:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely in English."},
        {"role": "user", "content": question},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
