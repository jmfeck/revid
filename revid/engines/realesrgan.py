# -*- coding: utf-8 -*-

"""Real-ESRGAN engine for AI upscaling.

Uses PyTorch CUDA for inference. Falls back to ncnn-vulkan binary if available.
"""

import glob
import os

from revid.engines.registry import register

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None

_MODEL_URLS = {
    "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "realesrgan-x4plus-anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}

_MODEL_CONFIGS = {
    "realesrgan-x4plus": {"num_block": 23, "scale": 4},
    "realesrgan-x4plus-anime": {"num_block": 6, "scale": 4},
}


@register("upscale", "realesrgan")
def upscale_realesrgan(step: dict, input_dir: str, output_dir: str) -> None:
    """Upscale frames using Real-ESRGAN."""
    factor = step.get("factor", 4)
    model_name = step.get("model", "realesrgan-x4plus")

    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise ImportError("Real-ESRGAN not found. Install with:\n  pip install realesrgan basicsr torch")

    config = _MODEL_CONFIGS.get(model_name, _MODEL_CONFIGS["realesrgan-x4plus"])
    model_url = _MODEL_URLS.get(model_name, _MODEL_URLS["realesrgan-x4plus"])

    network = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=config["num_block"],
        num_grow_ch=32,
        scale=config["scale"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    upsampler = RealESRGANer(
        scale=config["scale"],
        model_path=model_url,
        model=network,
        half=True if device == "cuda" else False,
        device=device,
    )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB"))
        img_bgr = img[:, :, ::-1].copy()
        output_bgr, _ = upsampler.enhance(img_bgr, outscale=factor)
        output_rgb = output_bgr[:, :, ::-1]
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(output_rgb).save(out_path)
