# -*- coding: utf-8 -*-

"""Colorization engines: DeOldify, DDColor, BigColor."""

import glob
import os

from revid.engines.registry import register


@register("colorize", "deoldify")
def colorize_deoldify(step: dict, input_dir: str, output_dir: str) -> None:
    """Colorize frames using DeOldify."""
    try:
        from deoldify import device as deoldify_device
        from deoldify.visualize import get_video_colorizer
    except ImportError:
        raise ImportError(
            "DeOldify not found. Install with:\n  pip install deoldify\n  Or clone https://github.com/jantic/DeOldify"
        )

    render_factor = step.get("render_factor", 35)

    colorizer = get_video_colorizer()

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        result = colorizer.get_transformed_image(
            frame_path,
            render_factor=render_factor,
            watermarked=False,
        )
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        result.save(out_path)


@register("colorize", "ddcolor")
def colorize_ddcolor(step: dict, input_dir: str, output_dir: str) -> None:
    """Colorize frames using DDColor."""
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError("DDColor not found. Install with:\n  pip install torch opencv-python")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = step.get("input_size", 512)

    try:
        from basicsr.utils.download_util import load_file_from_url
        from ddcolor.ddcolor_model import DDColor

        model_path = load_file_from_url(
            url="https://github.com/piddnad/DDColor/releases/download/v1.0/ddcolor_modelscope.pth",
            model_dir="weights/DDColor",
            progress=True,
        )

        model = DDColor(input_size=input_size).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("params", checkpoint))
        model.eval()
    except ImportError:
        raise ImportError("DDColor model not found. Clone from:\n  https://github.com/piddnad/DDColor")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = cv2.imread(frame_path)
        orig_h, orig_w = img.shape[:2]

        # Convert to LAB, extract L channel
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = img_lab[:, :, 0]

        # Resize for model
        img_resized = cv2.resize(img, (input_size, input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_l = torch.from_numpy(img_l.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 50.0 - 1.0

        with torch.no_grad():
            ab = model(img_l)

        # Resize ab back to original size
        ab = ab[0].permute(1, 2, 0).cpu().numpy() * 110.0
        ab = cv2.resize(ab, (orig_w, orig_h))

        # Merge L + ab
        result_lab = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        result_lab[:, :, 0] = l_channel
        result_lab[:, :, 1:] = ab
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, result_bgr)


@register("colorize", "bigcolor")
def colorize_bigcolor(step: dict, input_dir: str, output_dir: str) -> None:
    """Colorize frames using BigColor."""
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError:
        raise ImportError(
            "BigColor not found. Install with:\n  pip install torch\n  Clone https://github.com/KIMGEONUNG/BigColor"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from bigcolor.model import BigColorModel

        model = BigColorModel().to(device)
        model.eval()
    except ImportError:
        raise ImportError("BigColor model not found. Clone from:\n  https://github.com/KIMGEONUNG/BigColor")

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("L"))
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            output = model(tensor)

        out_img = (output[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(out_img).save(out_path)
