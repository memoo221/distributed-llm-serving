from functools import lru_cache

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from master.routers.mock_api import router as mock_router


# MODEL_NAME = "Qwen/Qwen2.5-0.5B"

app = FastAPI(title="Distributed LLM Serving API")
router = APIRouter()

app.include_router(router)
app.include_router(mock_router, prefix="/mock")


# class GenerateRequest(BaseModel):
# 	prompt: str = Field(..., min_length=1)
# 	max_new_tokens: int = Field(default=128, ge=1, le=512)


# class GenerateResponse(BaseModel):
# 	response: str
# 	metrics: dict


# def _bytes_to_mb(num_bytes: int) -> float:
# 	return round(num_bytes / (1024 * 1024), 2)


# def get_vram_stats() -> dict:
# 	if not torch.cuda.is_available():
# 		return {
# 			"device": "cpu",
# 			"gpu_used_mb": 0.0,
# 		}

# 	device = torch.cuda.current_device()
# 	allocated = torch.cuda.memory_allocated(device)

# 	return {
# 		"device": f"cuda:{device}",
# 		"gpu_used_mb": _bytes_to_mb(allocated),
# 	}


# def diff_vram_stats(before: dict, after: dict) -> dict:
# 	if before["device"] == "cpu" or after["device"] == "cpu":
# 		return {
# 			"gpu_taken_mb": 0.0,
# 		}

# 	return {
# 		"gpu_taken_mb": round(after["gpu_used_mb"] - before["gpu_used_mb"], 2),
# 	}


# @lru_cache(maxsize=1)
# def get_model_bundle():
# 	before_load = get_vram_stats()
# 	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 	model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 	if torch.cuda.is_available():
# 		model = model.to("cuda")
# 		torch.cuda.synchronize()

# 	after_load = get_vram_stats()
# 	load_metrics = {
# 		"device": after_load["device"],
# 		"gpu_taken_mb": diff_vram_stats(before_load, after_load)["gpu_taken_mb"],
# 		"gpu_used_after_load_mb": after_load["gpu_used_mb"],
# 	}
# 	return tokenizer, model, load_metrics


# @app.get("/health")
# def health_check():
# 	return {"status": "ok"}


# @router.post("/generate", response_model=GenerateResponse)
# def generate_text(request: GenerateRequest):
    
# 	try:
# 		tokenizer, model, load_metrics = get_model_bundle()
# 		request_before = get_vram_stats()
# 		inputs = tokenizer(request.prompt, return_tensors="pt")
    

# 		if torch.cuda.is_available():
# 			inputs = {key: value.to("cuda") for key, value in inputs.items()}

# 		output_tokens = model.generate(
# 			**inputs,
# 			max_new_tokens=request.max_new_tokens,
# 			do_sample=False,
# 		)
# 		if torch.cuda.is_available():
# 			torch.cuda.synchronize()
# 		request_after = get_vram_stats()

# 		generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
# 		return GenerateResponse(
# 			response=generated_text,
# 			metrics={
# 				"model_load_vram": load_metrics,
# 				"request_vram": {
# 					"device": request_after["device"],
# 					"gpu_taken_mb": diff_vram_stats(request_before, request_after)["gpu_taken_mb"],
# 					"gpu_used_before_request_mb": request_before["gpu_used_mb"],
# 					"gpu_used_after_request_mb": request_after["gpu_used_mb"],
# 				},
# 			},
# 		)
# 	except Exception as exc:
# 		raise HTTPException(status_code=500, detail=f"Model generation failed: {exc}") from exc




if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
