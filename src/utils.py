import os
from typing import Any, Dict, Tuple

import torch


def get_device_config() -> Tuple[int, int, str]:
    """Determines GPU configuration and distribution strategy."""
    n_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    devices = torch.cuda.device_count()
    strategy = "ddp" if n_gpus > 1 else "auto"
    return n_gpus, devices, strategy

def load_vae_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """Loads and filters VAE weights from a checkpoint file."""
    print(f"[INFO] Loading VAE weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    
    # Filter for first_stage_model keys
    return {
        k.replace("first_stage_model.", ""): v 
        for k, v in state.items() 
        if "first_stage_model" in k
    }