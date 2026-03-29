# -*- coding: utf-8 -*-

"""Inpainting and object removal engines: LaMa, MAT, ProPainter, E2FGVI."""

import glob
import os

import numpy as np
from PIL import Image

from revid.engines.registry import register


def _load_mask(mask_path: str, h: int, w: int) -> np.ndarray:
    """Load and resize a binary mask to match frame dimensions."""
    mask = np.array(Image.open(mask_path).convert("L").resize((w, h)))
    return (mask > 127).astype(np.float32)


# =============================================================================
# Per-frame inpainting
# =============================================================================


@register("inpaint", "lama")
def inpaint_lama(step: dict, input_dir: str, output_dir: str) -> None:
    """Inpaint damaged regions using LaMa (Large Mask Inpainting)."""
    try:
        import torch
    except ImportError:
        raise ImportError("LaMa requires PyTorch. Install with: pip install torch")

    mask_path = step.get("mask_path")
    if not mask_path:
        raise ValueError("inpaint requires a mask_path in step params")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from simple_lama_inpainting.inpaint import SimpleLama

        model = SimpleLama()
    except ImportError:
        try:
            from basicsr.utils.download_util import load_file_from_url
            from lama.saicinpainting.training.trainers import load_checkpoint

            model_path = load_file_from_url(
                url="https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
                model_dir="weights/LaMa",
                progress=True,
            )
            model = torch.load(model_path, map_location=device)
            model.eval()
        except ImportError:
            raise ImportError(
                "LaMa not found. Install with:\n"
                "  pip install simple-lama-inpainting\n"
                "  Or clone https://github.com/advimman/lama"
            )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = Image.open(frame_path).convert("RGB")
        w, h = img.size
        mask = Image.open(mask_path).convert("L").resize((w, h))

        try:
            result = model(img, mask)
        except TypeError:
            # Fallback for torch model
            img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            mask_t = torch.from_numpy(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                result_t = model(img_t, mask_t)
            result = Image.fromarray((result_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        result.save(out_path)


@register("inpaint", "mat")
def inpaint_mat(step: dict, input_dir: str, output_dir: str) -> None:
    """Inpaint damaged regions using MAT (Mask-Aware Transformer)."""
    try:
        import torch
    except ImportError:
        raise ImportError("MAT requires PyTorch. Install with: pip install torch")

    mask_path = step.get("mask_path")
    if not mask_path:
        raise ValueError("inpaint requires a mask_path in step params")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from basicsr.utils.download_util import load_file_from_url
        from mat.networks.mat import Generator

        model_path = load_file_from_url(
            url="https://github.com/fenglinglwb/MAT/releases/download/v1.0/Places_512_FullData.pkl",
            model_dir="weights/MAT",
            progress=True,
        )

        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)["G_ema"].to(device)
        model.eval()
    except ImportError:
        raise ImportError("MAT not found. Clone from:\n  https://github.com/fenglinglwb/MAT")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB"))
        h, w = img.shape[:2]
        mask = _load_mask(mask_path, h, w)

        img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t, mask_t, truncation_psi=1)

        out_img = ((output[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(out_img).save(out_path)


# =============================================================================
# Video-aware object removal
# =============================================================================


@register("object_remove", "propainter")
def object_remove_propainter(step: dict, input_dir: str, output_dir: str) -> None:
    """Remove objects from video using ProPainter (flow-guided video inpainting)."""
    try:
        import torch
    except ImportError:
        raise ImportError("ProPainter requires PyTorch. Install with: pip install torch")

    mask_path = step.get("mask_path")
    if not mask_path:
        raise ValueError("object_remove requires a mask_path in step params")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from basicsr.utils.download_util import load_file_from_url
        from propainter.core.propainter import ProPainter

        model_path = load_file_from_url(
            url="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
            model_dir="weights/ProPainter",
            progress=True,
        )
        flow_model_path = load_file_from_url(
            url="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
            model_dir="weights/ProPainter",
            progress=True,
        )

        model = ProPainter(model_path=model_path, flow_model_path=flow_model_path).to(device)
        model.eval()
    except ImportError:
        raise ImportError("ProPainter not found. Clone from:\n  https://github.com/sczhou/ProPainter")

    # Load all frames and mask
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    imgs = []
    for fp in frames:
        img = np.array(Image.open(fp).convert("RGB")).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))

    video_tensor = torch.stack(imgs).unsqueeze(0).to(device)
    h, w = imgs[0].shape[1], imgs[0].shape[2]
    mask = _load_mask(mask_path, h, w)
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).expand(-1, len(frames), -1, -1).to(device)

    with torch.no_grad():
        output = model(video_tensor, mask_tensor)

    for i, fp in enumerate(frames):
        out_img = (output[0, i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, os.path.basename(fp))
        Image.fromarray(out_img).save(out_path)


@register("object_remove", "e2fgvi")
def object_remove_e2fgvi(step: dict, input_dir: str, output_dir: str) -> None:
    """Remove objects from video using E2FGVI (flow-guided video completion)."""
    try:
        import torch
    except ImportError:
        raise ImportError("E2FGVI requires PyTorch. Install with: pip install torch")

    mask_path = step.get("mask_path")
    if not mask_path:
        raise ValueError("object_remove requires a mask_path in step params")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from basicsr.utils.download_util import load_file_from_url
        from e2fgvi.model.e2fgvi import InpaintGenerator

        model_path = load_file_from_url(
            url="https://github.com/MCG-NKU/E2FGVI/releases/download/v1.0/E2FGVI_HQ.pth",
            model_dir="weights/E2FGVI",
            progress=True,
        )

        model = InpaintGenerator().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("generator", checkpoint))
        model.eval()
    except ImportError:
        raise ImportError("E2FGVI not found. Clone from:\n  https://github.com/MCG-NKU/E2FGVI")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    imgs = []
    for fp in frames:
        img = np.array(Image.open(fp).convert("RGB")).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(img).permute(2, 0, 1))

    video_tensor = torch.stack(imgs).unsqueeze(0).to(device)
    h, w = imgs[0].shape[1], imgs[0].shape[2]
    mask = _load_mask(mask_path, h, w)
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).expand(-1, len(frames), -1, -1).to(device)

    with torch.no_grad():
        output = model(video_tensor, mask_tensor)

    for i, fp in enumerate(frames):
        out_img = (output[0, i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, os.path.basename(fp))
        Image.fromarray(out_img).save(out_path)
