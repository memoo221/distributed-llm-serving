"""
Downloads Qwen/Qwen2.5-0.5B from HuggingFace into ./qwen2.5-0.5b
"""
from huggingface_hub import snapshot_download
import os

MODEL_ID = "Qwen/Qwen2.5-0.5B"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "qwen2.5-0.5b")

print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)


print("Done. Model saved to:", LOCAL_DIR)
