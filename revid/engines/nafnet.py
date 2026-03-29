# -*- coding: utf-8 -*-

"""Denoise and deblur engines: NAFNet, SCUNet, Restormer, MPRNet, HINet."""

import glob
import os

import numpy as np
from PIL import Image

from revid.engines.registry import register


def _setup_torch():
    """Common torch setup."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch, device


def _process_frames(model, input_dir: str, output_dir: str, tile_size: int = 0):
    """Process all frames through a model with optional tiling."""
    import torch

    device = next(model.parameters()).device
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad to multiple of required size
        _, _, h, w = tensor.shape
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            output = model(tensor)

        # Crop padding
        output = output[:, :, :h, :w]
        out_img = (output[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(out_img).save(out_path)


# =============================================================================
# Denoise
# =============================================================================

@register("denoise", "nafnet")
def denoise_nafnet(step: dict, input_dir: str, output_dir: str) -> None:
    """Denoise frames using NAFNet."""
    try:
        torch, device = _setup_torch()
        from basicsr.archs.nafnet_arch import NAFNet
    except ImportError:
        raise ImportError(
            "NAFNet not found. Install with:\n"
            "  pip install basicsr torch"
        )

    from basicsr.utils.download_util import load_file_from_url

    model_path = load_file_from_url(
        url="https://github.com/megvii-research/NAFNet/releases/download/v0.1.0/NAFNet-SIDD-width64.pth",
        model_dir="weights/NAFNet",
        progress=True,
    )

    model = NAFNet(
        img_channel=3, width=64,
        enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
        middle_blk_num=12,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("params", checkpoint))
    model.eval()

    _process_frames(model, input_dir, output_dir)


@register("denoise", "scunet")
def denoise_scunet(step: dict, input_dir: str, output_dir: str) -> None:
    """Denoise frames using SCUNet (Swin-Conv-UNet)."""
    try:
        torch, device = _setup_torch()
    except ImportError:
        raise ImportError("SCUNet requires PyTorch. Install with: pip install torch")

    try:
        from scunet.models.network_scunet import SCUNet as SCUNetModel
    except ImportError:
        raise ImportError(
            "SCUNet not found. Clone from:\n"
            "  https://github.com/cszn/SCUNet"
        )

    model_url = "https://github.com/cszn/SCUNet/releases/download/v1.0/scunet_color_real_psnr.pth"
    from basicsr.utils.download_util import load_file_from_url
    model_path = load_file_from_url(url=model_url, model_dir="weights/SCUNet", progress=True)

    model = SCUNetModel(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _process_frames(model, input_dir, output_dir)


@register("denoise", "restormer")
def denoise_restormer(step: dict, input_dir: str, output_dir: str) -> None:
    """Denoise frames using Restormer (Transformer-based)."""
    try:
        torch, device = _setup_torch()
    except ImportError:
        raise ImportError("Restormer requires PyTorch. Install with: pip install torch")

    try:
        from restormer.basicsr.models.archs.restormer_arch import Restormer
    except ImportError:
        raise ImportError(
            "Restormer not found. Clone from:\n"
            "  https://github.com/swz30/Restormer"
        )

    from basicsr.utils.download_util import load_file_from_url
    model_path = load_file_from_url(
        url="https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth",
        model_dir="weights/Restormer",
        progress=True,
    )

    model = Restormer(
        inp_channels=3, out_channels=3, dim=48,
        num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type="BiasFree",
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("params", checkpoint))
    model.eval()

    _process_frames(model, input_dir, output_dir)


# =============================================================================
# Deblur
# =============================================================================

@register("deblur", "nafnet")
def deblur_nafnet(step: dict, input_dir: str, output_dir: str) -> None:
    """Deblur frames using NAFNet (GoPro weights)."""
    try:
        torch, device = _setup_torch()
        from basicsr.archs.nafnet_arch import NAFNet
    except ImportError:
        raise ImportError(
            "NAFNet not found. Install with:\n"
            "  pip install basicsr torch"
        )

    from basicsr.utils.download_util import load_file_from_url

    model_path = load_file_from_url(
        url="https://github.com/megvii-research/NAFNet/releases/download/v0.1.0/NAFNet-GoPro-width64.pth",
        model_dir="weights/NAFNet",
        progress=True,
    )

    model = NAFNet(
        img_channel=3, width=64,
        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
        middle_blk_num=1,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("params", checkpoint))
    model.eval()

    _process_frames(model, input_dir, output_dir)


@register("deblur", "mprnet")
def deblur_mprnet(step: dict, input_dir: str, output_dir: str) -> None:
    """Deblur frames using MPRNet."""
    try:
        torch, device = _setup_torch()
    except ImportError:
        raise ImportError("MPRNet requires PyTorch. Install with: pip install torch")

    try:
        from mprnet.MPRNet import MPRNet
    except ImportError:
        raise ImportError(
            "MPRNet not found. Clone from:\n"
            "  https://github.com/swz30/MPRNet"
        )

    from basicsr.utils.download_util import load_file_from_url
    model_path = load_file_from_url(
        url="https://github.com/swz30/MPRNet/releases/download/v1.0/model_deblurring.pth",
        model_dir="weights/MPRNet",
        progress=True,
    )

    model = MPRNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    _process_frames(model, input_dir, output_dir)


@register("deblur", "hinet")
def deblur_hinet(step: dict, input_dir: str, output_dir: str) -> None:
    """Deblur frames using HINet."""
    try:
        torch, device = _setup_torch()
    except ImportError:
        raise ImportError("HINet requires PyTorch. Install with: pip install torch")

    try:
        from hinet.HINet import HINet
    except ImportError:
        raise ImportError(
            "HINet not found. Clone from:\n"
            "  https://github.com/megvii-model/HINet"
        )

    from basicsr.utils.download_util import load_file_from_url
    model_path = load_file_from_url(
        url="https://github.com/megvii-model/HINet/releases/download/v1.0/HINet-GoPro.pth",
        model_dir="weights/HINet",
        progress=True,
    )

    model = HINet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("params", checkpoint))
    model.eval()

    _process_frames(model, input_dir, output_dir)
