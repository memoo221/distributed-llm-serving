import asyncio
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# # Try to import GPU monitoring
# try:
#     import pynvml
#     pynvml.nvmlInit()
#     GPU_MONITORING_AVAILABLE = True
# except Exception as e:
#     print(f"[WorkerNode] GPU monitoring unavailable: {e}")
#     GPU_MONITORING_AVAILABLE = False


class WorkerNode:
    """
    Unified worker node that supports both GPU (Groq) and CPU execution.
    Handles model loading, inference, device management, and GPU monitoring.
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: Optional[str] = None,
        batch_mode: bool = False,
        max_batch_size: int = 8,
        remote_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        worker_id: Optional[str] = None,
    ):
        """
        Initialize the worker node.
        
        Args:
            model_path: Path to the model (HuggingFace model ID or local path)
            device: Device to use.
                - "cuda:0", "cuda:1", etc: Use GPU (routes to a remote Groq worker endpoint)
                - "cpu": Use CPU
                - None: Auto-detect (prefers GPU if available locally)
                NOTE: For GPU mode, explicitly set device="cuda:0" so we skip the
                      local CUDA availability check (the remote endpoint owns the GPU).
            batch_mode: Whether to enable batching
            max_batch_size: Maximum batch size for inference
            remote_endpoint: For GPU mode, the URL of the Groq worker endpoint
                           (e.g., "http://localhost:9001")
        """
        self.model_path = model_path
        self.batch_mode = batch_mode
        self.max_batch_size = max_batch_size
        self.remote_endpoint = remote_endpoint
        self.api_key = api_key or os.getenv("WORKER_API_KEY")
        self.worker_id = worker_id or os.getenv("WORKER_ID") or f"worker-{uuid.uuid4().hex[:8]}"

        # Heartbeat state
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop_event: Optional[threading.Event] = None
        self._hb_master_url: Optional[str] = None
        self._hb_interval_sec: float = 5.0
        
        # Device selection: auto-detect if not specified
        self.device = self._select_device(device)
        self.device_idx = self._get_device_idx()
        
        print(f"[WorkerNode] Using device: {self.device}")
        if "cuda" in self.device and self.remote_endpoint:
            print(f"[WorkerNode] GPU mode - will route to remote endpoint: {self.remote_endpoint}")
        
        # Load model and tokenizer only for CPU mode
        self.tokenizer = None
        self.model = None
        if "cpu" in self.device:
            self._load_model()
        else:
            print(f"[WorkerNode] GPU mode - model loading skipped (will use remote endpoint)")
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.started_at = time.time()

        # In-flight request counter — reported on heartbeat so the master
        # scheduler sees real load. Synchronized because /generate runs in
        # FastAPI's threadpool (sync route) and the heartbeat thread reads it.
        self.active_requests = 0
        self._active_lock = threading.Lock()
    
    def _select_device(self, device: Optional[str]) -> str:
        """
        Select device for inference.
        
        If device is explicitly specified (e.g., "cuda:0"), trust it without checking local CUDA.
        This is important when routing to a remote GPU worker (e.g., Groq).
        Only auto-detect when device is None.
        """
        if device is None:
            # Auto-detect: prefer GPU if available locally
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                return "cpu"
        
        # If device is explicitly specified, use it as-is (don't check local CUDA)
        # This allows tunneling to remote GPUs without local CUDA
        if device.lower().startswith("cuda"):
            return device
        
        return "cpu"
    
    def _inc_active(self, n: int = 1) -> None:
        with self._active_lock:
            self.active_requests += n

    def _dec_active(self, n: int = 1) -> None:
        with self._active_lock:
            self.active_requests -= n

    def _get_device_idx(self) -> Optional[int]:
        """Extract CUDA device index if applicable."""
        if "cuda" in self.device:
            try:
                return int(self.device.split(":")[1])
            except (IndexError, ValueError):
                return 0
        return None
    
    def _load_model(self) -> None:
        """Load tokenizer and model onto the selected device."""
        print(f"[WorkerNode] Loading model: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Ensure pad token is set (critical for batching)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            # Determine data type based on device
            dtype = torch.float16 if "cuda" in self.device else torch.float32
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
            ).to(self.device)
            self.model.eval()
            
            # Log memory usage if GPU
            if "cuda" in self.device:
                torch.cuda.synchronize(self.device)
                vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                print(f"[WorkerNode] Model loaded. VRAM: {vram_mb:.1f} MB")
        
        except Exception as e:
            print(f"[WorkerNode] Error loading model: {e}")
            raise
        
    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely in English."},
            {"role": "user", "content": question},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt
    
    
    def generate_single(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.5,
        top_p: float = 0.8,
        **kwargs
    ) -> str:
        """
        Generate text for a single prompt (local CPU inference).
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Use CPU device for local inference.")
        
        self.total_requests += 1
        
        try:    
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Extract only new tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            return response
        
        except Exception as e:
            self.total_errors += 1
            raise RuntimeError(f"Generation error: {e}")
    
    def _remote_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _call_remote_endpoint(
        self, prompt: str, max_new_tokens: int = 256, client: Optional[httpx.AsyncClient] = None
    ) -> str:
        """Call the remote Groq worker /generate endpoint. Reuses `client` if provided."""
        if not self.remote_endpoint:
            raise RuntimeError("Remote endpoint not configured for GPU mode")

        url = f"{self.remote_endpoint}/generate"
        payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
        headers = self._remote_headers()

        try:
            if client is not None:
                response = await client.post(url, json=payload, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=300.0) as c:
                    response = await c.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Remote endpoint error: {e}")

    async def _gather_remote(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Dispatch all prompts concurrently to the remote endpoint."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            tasks = [
                self._call_remote_endpoint(p, max_new_tokens, client=client)
                for p in prompts
            ]
            return await asyncio.gather(*tasks)

    def _call_remote_endpoint_sync(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Call the remote Groq worker /generate endpoint (synchronous)."""
        if not self.remote_endpoint:
            raise RuntimeError("Remote endpoint not configured for GPU mode")

        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{self.remote_endpoint}/generate",
                    json={"prompt": prompt, "max_new_tokens": max_new_tokens},
                    headers=self._remote_headers(),
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Remote endpoint error: {e}")
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text for a single prompt.
        
        Device-aware routing:
        - If device is CUDA: sends request to remote Groq worker endpoint
        - If device is CPU: runs inference locally using generate_single
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (ignored for GPU/remote mode)
            top_p: Top-p (nucleus) sampling parameter (ignored for GPU/remote mode)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        self._inc_active()
        try:
            if "cuda" in self.device:
                # GPU mode: route to remote endpoint
                print(f"[WorkerNode] GPU mode - routing to remote endpoint: {self.remote_endpoint}")
                self.total_requests += 1
                try:
                    return self._call_remote_endpoint_sync(prompt, max_new_tokens)
                except Exception:
                    self.total_errors += 1
                    raise
            else:
                # CPU mode: local inference
                full_prompt = self._build_prompt(prompt)
                return self.generate_single(full_prompt, max_new_tokens, temperature, top_p, **kwargs)
        finally:
            self._dec_active()
    
    def generate_concurrent(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Device-aware routing:
        - CUDA: dispatch all prompts concurrently to the remote Groq worker endpoint.
        - CPU:  process each prompt sequentially via generate_single.
                CPU has no batching benefit — padded batched generate is slower
                than sequential single calls (no early stop, no parallel matmul win).
        """
        n = len(prompts)
        self._inc_active(n)
        try:
            if "cuda" in self.device:
                print(f"[WorkerNode] GPU mode - dispatching {n} concurrent requests to remote endpoint")
                self.total_requests += n
                try:
                    return asyncio.run(self._gather_remote(prompts, max_new_tokens))
                except Exception:
                    self.total_errors += n
                    raise

            # CPU mode: sequential single-prompt inference. No batching.
            print(f"[WorkerNode] CPU mode - processing {n} prompts sequentially")
            return [
                self.generate_single(p, max_new_tokens, temperature, top_p, **kwargs)
                for p in prompts
            ]
        finally:
            self._dec_active(n)
    
    # def get_gpu_stats(self) -> Optional[Dict[str, Any]]:
    #     """Get GPU memory and utilization stats (if GPU and NVML available)."""
    #     if "cuda" not in self.device or self.device_idx is None or not GPU_MONITORING_AVAILABLE:
    #         return None
        
    #     try:
    #         handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)
    #         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #         util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
    #         return {
    #             "device": self.device,
    #             "vram_used_mb": mem_info.used / (1024 ** 2),
    #             "vram_total_mb": mem_info.total / (1024 ** 2),
    #             "vram_free_mb": mem_info.free / (1024 ** 2),
    #             "gpu_utilization": util.gpu,
    #             "memory_utilization": util.memory,
    #         }
    #     except Exception as e:
    #         print(f"[WorkerNode] Error getting GPU stats: {e}")
    #         return None
    
    def device_type(self) -> str:
        """Return coarse device type: 'cuda' or 'cpu' (drops the index)."""
        return "cuda" if "cuda" in self.device else "cpu"

    def _build_heartbeat_payload(self) -> Dict[str, Any]:
        """Payload sent to the master on each heartbeat. Must include
        active_requests so the master scheduler can balance load — without
        it the registry defaults the value to 0 and the worker always looks idle."""
        payload = self.get_stats()
        with self._active_lock:
            active = self.active_requests
        payload.update({
            "worker_id": self.worker_id,
            "device_type": self.device_type(),
            "active_requests": active,
            "queue_depth": 0,
            "remote_endpoint": self.remote_endpoint,
            "timestamp": time.time(),
        })
        return payload

    def _send_heartbeat(self, client: httpx.Client) -> None:
        """One-shot heartbeat POST. Errors are caught by the loop, not raised."""
        if not self._hb_master_url:
            return
        client.post(
            f"{self._hb_master_url.rstrip('/')}/heartbeat",
            json=self._build_heartbeat_payload(),
            headers=self._remote_headers(),
        ).raise_for_status()

    def _heartbeat_loop(self) -> None:
        """Background loop: send heartbeat every interval until stop event is set."""
        assert self._hb_stop_event is not None
        with httpx.Client(timeout=10.0) as client:
            while not self._hb_stop_event.is_set():
                try:
                    self._send_heartbeat(client)
                except Exception as e:
                    print(f"[WorkerNode] heartbeat failed: {type(e).__name__}: {e}")
                # wait() returns True if stop was set during the wait — exits cleanly
                if self._hb_stop_event.wait(self._hb_interval_sec):
                    break

    def start_heartbeat(self, master_url: str, interval_sec: float = 5.0) -> None:
        """Start a background thread that POSTs heartbeats to the master every interval."""
        if self._hb_thread is not None and self._hb_thread.is_alive():
            print("[WorkerNode] heartbeat already running")
            return
        self._hb_master_url = master_url
        self._hb_interval_sec = interval_sec
        self._hb_stop_event = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"hb-{self.worker_id}",
            daemon=True,
        )
        self._hb_thread.start()
        print(f"[WorkerNode] heartbeat started → {master_url} every {interval_sec}s")

    def stop_heartbeat(self, timeout_sec: float = 2.0) -> None:
        """Stop the heartbeat loop and join the thread."""
        if self._hb_stop_event is not None:
            self._hb_stop_event.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=timeout_sec)
        self._hb_thread = None
        self._hb_stop_event = None

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        uptime = time.time() - self.started_at

        stats = {
            "worker_id": self.worker_id,
            "device": self.device,
            "device_type": self.device_type(),
            "model": self.model_path,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "uptime_seconds": uptime,
        }
        
        # Add GPU stats if available
        # gpu_stats = self.get_gpu_stats()
        # if gpu_stats:
        #     stats.update(gpu_stats)
        
        return stats


# Backward compatibility: legacy WorkerService name
class WorkerService(WorkerNode):
    """Legacy alias for WorkerNode (for backward compatibility)."""
    pass










# _tokenizer = None
# _model = None
# _device = None


# def load_model(model_path: str):
#     global _tokenizer, _model, _device
#     _device = "cuda" if torch.cuda.is_available() else "cpu"
#     _tokenizer = AutoTokenizer.from_pretrained(model_path)
#     _model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
#     ).to(_device)
#     _model.eval()


# def generate_response(question: str, max_new_tokens: int = 256) -> str:
#     if _model is None or _tokenizer is None:
#         raise RuntimeError("Model not loaded. Call load_model() first.")

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely in English."},
#         {"role": "user", "content": question},
#     ]
#     prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
#     with torch.no_grad():
#         output_ids = _model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             pad_token_id=_tokenizer.eos_token_id,
#         )
#     new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
#     return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
