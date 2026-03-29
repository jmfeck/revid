# -*- coding: utf-8 -*-

"""Extra upscale engines: SwinIR, ESPCN, EDSR, BasicVSR++."""

import glob
import os

import numpy as np
from PIL import Image

from revid.engines.registry import register


@register("upscale", "swinir")
def upscale_swinir(step: dict, input_dir: str, output_dir: str) -> None:
    """Upscale frames using SwinIR (Swin Transformer)."""
    try:
        import torch
    except ImportError:
        raise ImportError("SwinIR requires PyTorch. Install with: pip install torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor = step.get("factor", 4)

    try:
        from basicsr.utils.download_util import load_file_from_url
        from swinir.models.network_swinir import SwinIR as SwinIRModel

        model_path = load_file_from_url(
            url="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
            "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-M_x4_GAN.pth",
            model_dir="weights/SwinIR",
            progress=True,
        )

        model = SwinIRModel(
            upscale=factor,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="1conv",
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        params = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        model.load_state_dict(params)
        model.eval()
    except ImportError:
        raise ImportError("SwinIR model not found. Clone from:\n  https://github.com/JingyunLiang/SwinIR")

    window_size = 8
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        _, _, h, w = tensor.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            output = model(tensor)

        output = output[:, :, : h * factor, : w * factor]
        out_img = (output[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(out_img).save(out_path)


@register("upscale", "espcn")
def upscale_espcn(step: dict, input_dir: str, output_dir: str) -> None:
    """Upscale frames using ESPCN (OpenCV DNN — fast, lightweight)."""
    try:
        import cv2

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
    except (ImportError, AttributeError):
        raise ImportError("ESPCN requires opencv-contrib-python. Install with:\n  pip install opencv-contrib-python")

    factor = step.get("factor", 4)

    model_url = f"https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x{factor}.pb"
    model_dir = os.path.join("weights", "ESPCN")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"ESPCN_x{factor}.pb")

    if not os.path.isfile(model_path):
        import urllib.request

        urllib.request.urlretrieve(model_url, model_path)

    sr.readModel(model_path)
    sr.setModel("espcn", factor)

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = cv2.imread(frame_path)
        result = sr.upsample(img)
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, result)


@register("upscale", "edsr")
def upscale_edsr(step: dict, input_dir: str, output_dir: str) -> None:
    """Upscale frames using EDSR (OpenCV DNN — higher quality than ESPCN)."""
    try:
        import cv2

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
    except (ImportError, AttributeError):
        raise ImportError("EDSR requires opencv-contrib-python. Install with:\n  pip install opencv-contrib-python")

    factor = step.get("factor", 4)

    model_url = f"https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x{factor}.pb"
    model_dir = os.path.join("weights", "EDSR")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"EDSR_x{factor}.pb")

    if not os.path.isfile(model_path):
        import urllib.request

        urllib.request.urlretrieve(model_url, model_path)

    sr.readModel(model_path)
    sr.setModel("edsr", factor)

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = cv2.imread(frame_path)
        result = sr.upsample(img)
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, result)


@register("upscale", "basicvsr")
def upscale_basicvsr(step: dict, input_dir: str, output_dir: str) -> None:
    """Upscale frames using BasicVSR++ (video SR — uses temporal info)."""
    try:
        import torch
    except ImportError:
        raise ImportError("BasicVSR++ requires PyTorch. Install with: pip install torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step.get("factor", 4)

    try:
        from basicsr.archs.basicvsr_arch import BasicVSR
        from basicsr.utils.download_util import load_file_from_url

        model_path = load_file_from_url(
            url="https://github.com/ckkelvinchan/BasicVSR_PlusPlus/releases/download/v1.0/BasicVSR_plusplus_REDS.pth",
            model_dir="weights/BasicVSR",
            progress=True,
        )

        model = BasicVSR(num_feat=64, num_block=30, spynet_path=None).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("params", checkpoint))
        model.eval()
    except ImportError:
        raise ImportError(
            "BasicVSR++ not found. Install with:\n"
            "  pip install basicsr\n"
            "  Or clone https://github.com/ckkelvinchan/BasicVSR_PlusPlus"
        )

    # Load all frames as a temporal sequence
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    batch_size = step.get("batch_size", 30)

    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]

        imgs = []
        for fp in batch_frames:
            img = np.array(Image.open(fp).convert("RGB")).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))

        # Stack as [1, T, C, H, W]
        sequence = torch.stack(imgs).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sequence)

        # Save each frame from output
        for i, fp in enumerate(batch_frames):
            out_img = (output[0, i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            out_path = os.path.join(output_dir, os.path.basename(fp))
            Image.fromarray(out_img).save(out_path)
