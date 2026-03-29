# -*- coding: utf-8 -*-

"""Frame interpolation engines: RIFE, IFRNet, AMT, FILM."""

import glob
import os
import shutil
import subprocess

from revid.engines.registry import register


def _find_ncnn_binary(name: str) -> str | None:
    """Find an ncnn binary in PATH or local tools."""
    found = shutil.which(name)
    if found:
        return found

    import revid

    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(revid.__file__)))
    for subdir in [".localtest/tools", "tools"]:
        for fname in [f"{name}.exe", name]:
            path = os.path.join(pkg_dir, subdir, name, fname)
            if os.path.isfile(path):
                return path
    return None


@register("interpolate", "rife")
def interpolate_rife(step: dict, input_dir: str, output_dir: str) -> None:
    """Interpolate frames using RIFE.

    Supports ncnn binary (preferred) or Python/PyTorch fallback.
    """
    multiplier = step.get("multiplier", 2)

    # Try ncnn binary first
    ncnn_bin = _find_ncnn_binary("rife-ncnn-vulkan")
    if ncnn_bin:
        models_dir = os.path.join(os.path.dirname(ncnn_bin), "models")
        cmd = [ncnn_bin, "-i", input_dir, "-o", output_dir]
        if multiplier != 2:
            cmd.extend(["-n", str(multiplier)])
        if os.path.isdir(models_dir):
            cmd.extend(["-m", os.path.join(models_dir, "rife-v4.6")])
        subprocess.run(cmd, check=True)
        return

    # Python fallback
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError(
            "RIFE not found. Either:\n  1. Download rife-ncnn-vulkan and add to PATH\n  2. pip install torch"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from rife.RIFE_HDv3 import Model

        model = Model()
        model.load_model(os.path.expanduser("~/.cache/revid/rife"), -1)
        model.eval()
    except ImportError:
        raise ImportError(
            "RIFE Python model not found. Install rife-ncnn-vulkan binary instead:\n"
            "  https://github.com/nihui/rife-ncnn-vulkan/releases"
        )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    out_idx = 0

    for i in range(len(frames) - 1):
        img0 = np.array(Image.open(frames[i]).convert("RGB")).astype(np.float32) / 255.0
        img1 = np.array(Image.open(frames[i + 1]).convert("RGB")).astype(np.float32) / 255.0

        t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(device)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)

        # Write original frame
        Image.open(frames[i]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
        out_idx += 1

        # Generate intermediate frames
        for m in range(1, multiplier):
            timestep = m / multiplier
            with torch.no_grad():
                mid = model.inference(t0, t1, timestep)
            mid_img = (mid[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mid_img).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
            out_idx += 1

    # Write last frame
    if frames:
        Image.open(frames[-1]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))


@register("interpolate", "ifrnet")
def interpolate_ifrnet(step: dict, input_dir: str, output_dir: str) -> None:
    """Interpolate frames using IFRNet."""
    ncnn_bin = _find_ncnn_binary("ifrnet-ncnn-vulkan")
    if ncnn_bin:
        cmd = [ncnn_bin, "-i", input_dir, "-o", output_dir]
        subprocess.run(cmd, check=True)
        return

    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError(
            "IFRNet not found. Either:\n"
            "  1. Download ifrnet-ncnn-vulkan\n"
            "  2. pip install torch and clone https://github.com/ltkong218/IFRNet"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiplier = step.get("multiplier", 2)

    try:
        from ifrnet.IFRNet import Model

        model = Model().to(device)
        model.eval()
    except ImportError:
        raise ImportError("IFRNet Python model not found. Install ifrnet-ncnn-vulkan instead.")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    out_idx = 0

    for i in range(len(frames) - 1):
        img0 = np.array(Image.open(frames[i]).convert("RGB")).astype(np.float32) / 255.0
        img1 = np.array(Image.open(frames[i + 1]).convert("RGB")).astype(np.float32) / 255.0

        t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(device)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)

        Image.open(frames[i]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
        out_idx += 1

        for m in range(1, multiplier):
            embt = torch.tensor([m / multiplier]).float().to(device)
            with torch.no_grad():
                mid = model(t0, t1, embt)
            mid_img = (mid[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mid_img).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
            out_idx += 1

    if frames:
        Image.open(frames[-1]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))


@register("interpolate", "amt")
def interpolate_amt(step: dict, input_dir: str, output_dir: str) -> None:
    """Interpolate frames using AMT (Accurate Motion Transfer)."""
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError("AMT not found. Install with:\n  pip install torch\n  Clone https://github.com/MCG-NKU/AMT")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiplier = step.get("multiplier", 2)
    model_size = step.get("model", "S")  # S, L, or G

    try:
        from amt.AMT_G import Model as AMT_G
        from amt.AMT_L import Model as AMT_L
        from amt.AMT_S import Model as AMT_S

        models = {"S": AMT_S, "L": AMT_L, "G": AMT_G}
        model = models[model_size]().to(device)
        model.eval()
    except ImportError:
        raise ImportError("AMT model not found. Clone from:\n  https://github.com/MCG-NKU/AMT")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    out_idx = 0

    for i in range(len(frames) - 1):
        img0 = np.array(Image.open(frames[i]).convert("RGB")).astype(np.float32) / 255.0
        img1 = np.array(Image.open(frames[i + 1]).convert("RGB")).astype(np.float32) / 255.0

        t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(device)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)

        Image.open(frames[i]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
        out_idx += 1

        for m in range(1, multiplier):
            timestep = torch.tensor([m / multiplier]).float().to(device)
            with torch.no_grad():
                mid = model(t0, t1, timestep)
            mid_img = (mid[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mid_img).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
            out_idx += 1

    if frames:
        Image.open(frames[-1]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))


@register("interpolate", "film")
def interpolate_film(step: dict, input_dir: str, output_dir: str) -> None:
    """Interpolate frames using FILM (Google)."""
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError(
            "FILM not found. Install with:\n"
            "  pip install torch\n"
            "  Clone https://github.com/google-research/frame-interpolation"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multiplier = step.get("multiplier", 2)

    try:
        from film_net.interpolator import Interpolator

        model = Interpolator().to(device)
        model.eval()
    except ImportError:
        raise ImportError("FILM model not found. Clone from:\n  https://github.com/google-research/frame-interpolation")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    out_idx = 0

    for i in range(len(frames) - 1):
        img0 = np.array(Image.open(frames[i]).convert("RGB")).astype(np.float32) / 255.0
        img1 = np.array(Image.open(frames[i + 1]).convert("RGB")).astype(np.float32) / 255.0

        t0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(device)
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)

        Image.open(frames[i]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
        out_idx += 1

        for m in range(1, multiplier):
            dt = m / multiplier
            with torch.no_grad():
                mid = model(t0, t1, dt)
            mid_img = (mid[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mid_img).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
            out_idx += 1

    if frames:
        Image.open(frames[-1]).save(os.path.join(output_dir, f"{out_idx:08d}.png"))
