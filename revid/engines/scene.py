# -*- coding: utf-8 -*-

"""Scene detection engines: PySceneDetect, TransNetV2."""

import os

from revid.engines.registry import register


@register("scene_detect", "pyscenedetect")
def scene_detect_pyscenedetect(step: dict, input_dir: str, output_dir: str) -> list[float]:
    """Detect scene boundaries using PySceneDetect."""
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:
        raise ImportError(
            "PySceneDetect not found. Install with:\n"
            "  pip install scenedetect[opencv]"
        )

    video_path = step.get("video_path")
    if not video_path:
        raise ValueError("scene_detect requires video_path in step params")

    threshold = step.get("threshold", 27.0)
    min_scene_len = step.get("min_scene_len", 15)

    scene_list = detect(
        video_path,
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len),
    )

    timestamps = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        scene[1].get_seconds()
        timestamps.append(start_time)

    # Write timestamps to output for pipeline consumption
    out_path = os.path.join(output_dir, "scenes.txt")
    with open(out_path, "w") as f:
        for ts in timestamps:
            f.write(f"{ts:.3f}\n")

    return timestamps


@register("scene_detect", "transnetv2")
def scene_detect_transnetv2(step: dict, input_dir: str, output_dir: str) -> list[float]:
    """Detect scene boundaries using TransNetV2."""
    try:
        import torch
        from transnetv2 import TransNetV2
    except ImportError:
        raise ImportError(
            "TransNetV2 not found. Install with:\n"
            "  pip install transnetv2"
        )

    import glob

    import numpy as np
    from PIL import Image

    model = TransNetV2()

    # Load frames
    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    imgs = []
    for fp in frames:
        img = np.array(Image.open(fp).convert("RGB"))
        # TransNetV2 expects 48x27 thumbnails
        img_resized = np.array(Image.fromarray(img).resize((48, 27)))
        imgs.append(img_resized)

    video = np.stack(imgs)  # [T, 27, 48, 3]

    predictions = model.predict_frames(video)
    scenes = model.predictions_to_scenes(predictions)

    fps = step.get("fps", 30.0)
    timestamps = [scene[0] / fps for scene in scenes]

    out_path = os.path.join(output_dir, "scenes.txt")
    with open(out_path, "w") as f:
        for ts in timestamps:
            f.write(f"{ts:.3f}\n")

    return timestamps
