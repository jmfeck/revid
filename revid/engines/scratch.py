# -*- coding: utf-8 -*-

"""Scratch and damage removal engines: RTN, Old Photo Restore."""

import glob
import os

import numpy as np
from PIL import Image

from revid.engines.registry import register


@register("scratch_remove", "rtn")
def scratch_remove_rtn(step: dict, input_dir: str, output_dir: str) -> None:
    """Remove scratches and tape damage using RTN / Bringing Old Films Back to Life."""
    try:
        import torch
    except ImportError:
        raise ImportError("RTN requires PyTorch. Install with: pip install torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from basicsr.utils.download_util import load_file_from_url
        from old_films.models import RestoreNet, ScratchDetector

        det_path = load_file_from_url(
            url="https://github.com/raywzy/Bringing-Old-Films-Back-to-Life/releases/download/v1.0/scratch_detector.pth",
            model_dir="weights/RTN",
            progress=True,
        )
        restore_path = load_file_from_url(
            url="https://github.com/raywzy/Bringing-Old-Films-Back-to-Life/releases/download/v1.0/restore_net.pth",
            model_dir="weights/RTN",
            progress=True,
        )

        detector = ScratchDetector().to(device)
        detector.load_state_dict(torch.load(det_path, map_location=device))
        detector.eval()

        restorer = RestoreNet().to(device)
        restorer.load_state_dict(torch.load(restore_path, map_location=device))
        restorer.eval()
    except ImportError:
        raise ImportError("RTN not found. Clone from:\n  https://github.com/raywzy/Bringing-Old-Films-Back-to-Life")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Stage 1: detect scratches
        with torch.no_grad():
            scratch_mask = detector(tensor)

        # Stage 2: restore using detected mask
        with torch.no_grad():
            output = restorer(tensor, scratch_mask)

        out_img = (output[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(out_img).save(out_path)


@register("scratch_remove", "old_photo_restore")
def scratch_remove_old_photo(step: dict, input_dir: str, output_dir: str) -> None:
    """Remove damage using Bringing Old Photos Back to Life (Microsoft)."""
    try:
        import torch
    except ImportError:
        raise ImportError("Old Photo Restore requires PyTorch. Install with: pip install torch")

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from old_photos.run import process_image
    except ImportError:
        raise ImportError(
            "Old Photo Restore not found. Clone from:\n  https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life"
        )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        result = process_image(
            frame_path,
            with_scratch=True,
            gpu_id=0 if torch.cuda.is_available() else -1,
        )
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        if isinstance(result, np.ndarray):
            Image.fromarray(result).save(out_path)
        else:
            result.save(out_path)
