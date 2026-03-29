# -*- coding: utf-8 -*-

"""AI stabilization engines: RAFT, FlowFormer.

These compute dense optical flow, estimate global motion, smooth the trajectory,
and warp frames to produce stabilized output.
"""

import glob
import os

import numpy as np
from PIL import Image

from revid.engines.registry import register


def _stabilize_with_flow_model(model, input_dir: str, output_dir: str, smoothing: int = 30):
    """Common stabilization pipeline using an optical flow model."""
    import cv2
    import torch

    device = next(model.parameters()).device
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    if len(frames) < 2:
        for fp in frames:
            Image.open(fp).save(os.path.join(output_dir, os.path.basename(fp)))
        return

    # Step 1: Compute optical flow between consecutive frames
    transforms = []
    for i in range(len(frames) - 1):
        img0 = np.array(Image.open(frames[i]).convert("RGB")).astype(np.float32)
        img1 = np.array(Image.open(frames[i + 1]).convert("RGB")).astype(np.float32)

        t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            flow = model(t0, t1)[-1][0].permute(1, 2, 0).cpu().numpy()

        # Estimate global translation from median flow
        dx = np.median(flow[:, :, 0])
        dy = np.median(flow[:, :, 1])
        transforms.append((dx, dy))

    # Step 2: Compute cumulative trajectory
    trajectory_x = np.cumsum([0] + [t[0] for t in transforms])
    trajectory_y = np.cumsum([0] + [t[1] for t in transforms])

    # Step 3: Smooth trajectory
    from scipy.ndimage import uniform_filter1d

    smooth_x = uniform_filter1d(trajectory_x, size=smoothing)
    smooth_y = uniform_filter1d(trajectory_y, size=smoothing)

    # Step 4: Compute correction
    correction_x = smooth_x - trajectory_x
    correction_y = smooth_y - trajectory_y

    # Step 5: Warp each frame
    for i, fp in enumerate(frames):
        img = cv2.imread(fp)
        h, w = img.shape[:2]

        M = np.float32([[1, 0, correction_x[i]], [0, 1, correction_y[i]]])
        stabilized = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        out_path = os.path.join(output_dir, os.path.basename(fp))
        cv2.imwrite(out_path, stabilized)


@register("stabilize", "raft")
def stabilize_raft(step: dict, input_dir: str, output_dir: str) -> None:
    """Stabilize video using RAFT optical flow."""
    try:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError:
        raise ImportError("RAFT requires PyTorch + torchvision. Install with:\n  pip install torch torchvision")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoothing = step.get("smoothing", 30)

    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
    model.eval()

    _stabilize_with_flow_model(model, input_dir, output_dir, smoothing=smoothing)


@register("stabilize", "flowformer")
def stabilize_flowformer(step: dict, input_dir: str, output_dir: str) -> None:
    """Stabilize video using FlowFormer optical flow."""
    try:
        import torch
    except ImportError:
        raise ImportError("FlowFormer requires PyTorch. Install with: pip install torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoothing = step.get("smoothing", 30)

    try:
        from basicsr.utils.download_util import load_file_from_url
        from flowformer.FlowFormer import FlowFormer

        model_path = load_file_from_url(
            url="https://github.com/drinkingcoder/FlowFormer-Official/releases/download/v1.0/flowformer-things.pth",
            model_dir="weights/FlowFormer",
            progress=True,
        )

        model = FlowFormer().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    except ImportError:
        raise ImportError("FlowFormer not found. Clone from:\n  https://github.com/drinkingcoder/FlowFormer-Official")

    _stabilize_with_flow_model(model, input_dir, output_dir, smoothing=smoothing)
