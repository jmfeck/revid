# -*- coding: utf-8 -*-

"""Face restoration engines: GFPGAN, CodeFormer, RestoreFormer."""

import glob
import os

from revid.engines.registry import register


@register("face_restore", "gfpgan")
def face_restore_gfpgan(step: dict, input_dir: str, output_dir: str) -> None:
    """Restore faces using GFPGAN."""
    try:
        from gfpgan import GFPGANer
    except ImportError:
        raise ImportError(
            "GFPGAN not found. Install with:\n"
            "  pip install gfpgan facexlib"
        )

    import numpy as np
    from PIL import Image

    upscale = step.get("upscale", 1)

    restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
    )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB"))
        img_bgr = img[:, :, ::-1].copy()
        _, _, output_bgr = restorer.enhance(img_bgr, paste_back=True)
        output_rgb = output_bgr[:, :, ::-1]
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(output_rgb).save(out_path)


@register("face_restore", "codeformer")
def face_restore_codeformer(step: dict, input_dir: str, output_dir: str) -> None:
    """Restore faces using CodeFormer."""
    try:
        import torch
        from basicsr.utils import img2tensor, tensor2img
        from basicsr.utils.download_util import load_file_from_url
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from torchvision.transforms.functional import normalize
    except ImportError:
        raise ImportError(
            "CodeFormer not found. Install with:\n"
            "  pip install codeformer-pip\n"
            "  Or clone https://github.com/sczhou/CodeFormer"
        )

    import numpy as np
    from PIL import Image

    fidelity = step.get("fidelity", 0.5)
    upscale = step.get("upscale", 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CodeFormer model
    model_path = load_file_from_url(
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        model_dir="weights/CodeFormer",
        progress=True,
    )

    from codeformer.basicsr.archs.codeformer_arch import CodeFormer
    model = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)["params_ema"]
    model.load_state_dict(checkpoint)
    model.eval()

    face_helper = FaceRestoreHelper(
        upscale, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png", device=device,
    )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB"))
        img_bgr = img[:, :, ::-1].copy()

        face_helper.clean_all()
        face_helper.read_image(img_bgr)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_t, w=fidelity)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        face_helper.get_inverse_affine(None)
        restored_bgr = face_helper.paste_faces_to_input_image()
        restored_rgb = restored_bgr[:, :, ::-1]
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(restored_rgb).save(out_path)


@register("face_restore", "restoreformer")
def face_restore_restoreformer(step: dict, input_dir: str, output_dir: str) -> None:
    """Restore faces using RestoreFormer."""
    try:
        import torch
        from basicsr.utils.download_util import load_file_from_url
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
    except ImportError:
        raise ImportError(
            "RestoreFormer not found. Install with:\n"
            "  pip install facexlib basicsr\n"
            "  Download weights from https://github.com/wzhouxiff/RestoreFormer"
        )

    import numpy as np
    from PIL import Image

    upscale = step.get("upscale", 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = load_file_from_url(
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        model_dir="weights/RestoreFormer",
        progress=True,
    )


    FaceRestoreHelper(
        upscale, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png", device=device,
    )

    # Load RestoreFormer via GFPGANer wrapper (compatible interface)
    from gfpgan import GFPGANer
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch="RestoreFormer",
        channel_multiplier=2,
    )

    frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    for frame_path in frames:
        img = np.array(Image.open(frame_path).convert("RGB"))
        img_bgr = img[:, :, ::-1].copy()
        _, _, output_bgr = restorer.enhance(img_bgr, paste_back=True)
        output_rgb = output_bgr[:, :, ::-1]
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        Image.fromarray(output_rgb).save(out_path)
