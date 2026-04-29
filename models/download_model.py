"""
Downloads TinyLlama-1.1B-Chat-v1.0 from HuggingFace into ./tinyllama-1.1b-chat
"""
from huggingface_hub import snapshot_download
import os

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "tinyllama-1.1b-chat")

print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)



print("Done. Model saved to:", LOCAL_DIR)
