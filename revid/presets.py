# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from revid.video import VideoFile


def _vhs_standard(video: "VideoFile") -> "VideoFile":
    """Fast VHS restoration using FFmpeg filters only."""
    return (
        video
        .deinterlace()
        .denoise(strength=0.5)
        .color_correct(brightness=0.05, contrast=1.1, saturation=1.2)
        .sharpen(amount=0.8)
        .upscale(factor=2)
    )


def _vhs_quality(video: "VideoFile") -> "VideoFile":
    """Higher quality VHS restoration with stronger filtering."""
    return (
        video
        .deinterlace(algorithm="bwdif")
        .denoise(strength=0.6, algorithm="nlmeans")
        .deflicker()
        .color_correct(brightness=0.05, contrast=1.15, saturation=1.25, gamma=1.05)
        .color_normalize()
        .sharpen(amount=1.2)
        .upscale(factor=2)
    )


def _dvd_cleanup(video: "VideoFile") -> "VideoFile":
    """Clean up DVD rips — deblock, deband, light denoise."""
    return (
        video
        .deblock(strength=0.4)
        .deband()
        .denoise(strength=0.3)
        .sharpen(amount=0.5)
    )


def _camcorder(video: "VideoFile") -> "VideoFile":
    """Restore old camcorder footage (Hi8, MiniDV, etc.)."""
    return (
        video
        .deinterlace()
        .denoise(strength=0.4)
        .color_correct(brightness=0.03, contrast=1.1, saturation=1.15)
        .sharpen(amount=0.6)
        .upscale(factor=2)
    )


def _film_8mm(video: "VideoFile") -> "VideoFile":
    """Restore digitized 8mm / Super 8 film."""
    return (
        video
        .deflicker(size=7)
        .denoise(strength=0.5, algorithm="nlmeans")
        .color_correct(brightness=0.05, contrast=1.2, saturation=1.3, gamma=1.1)
        .sharpen(amount=1.0)
        .upscale(factor=2)
    )


# =========================================================================
# AI-powered presets (require revid[ai])
# =========================================================================


def _vhs_ai(video: "VideoFile") -> "VideoFile":
    """VHS restoration with AI upscaling and face restoration."""
    return (
        video
        .deinterlace(algorithm="bwdif")
        .denoise(strength=0.5, algorithm="nlmeans")
        .deflicker()
        .color_correct(brightness=0.05, contrast=1.15, saturation=1.2)
        .sharpen(amount=0.6)
        .upscale(factor=4, engine="realesrgan")
        .face_restore(engine="gfpgan")
    )


def _vhs_ai_full(video: "VideoFile") -> "VideoFile":
    """Maximum quality VHS restoration — full AI pipeline."""
    return (
        video
        .deinterlace(algorithm="bwdif")
        .denoise(engine="nafnet")
        .deflicker()
        .color_correct(brightness=0.05, contrast=1.15, saturation=1.25, gamma=1.05)
        .color_normalize()
        .upscale(factor=4, engine="realesrgan")
        .face_restore(engine="gfpgan")
        .interpolate(target_fps=60, engine="rife")
    )


def _camcorder_ai(video: "VideoFile") -> "VideoFile":
    """Camcorder restoration with AI upscaling and face restoration."""
    return (
        video
        .deinterlace()
        .denoise(strength=0.4)
        .color_correct(brightness=0.03, contrast=1.1, saturation=1.15)
        .upscale(factor=4, engine="realesrgan")
        .face_restore(engine="gfpgan")
    )


def _film_8mm_ai(video: "VideoFile") -> "VideoFile":
    """8mm film restoration with AI denoise, upscale, and scratch removal."""
    return (
        video
        .deflicker(size=7)
        .scratch_remove(engine="rtn")
        .denoise(engine="nafnet")
        .color_correct(brightness=0.05, contrast=1.2, saturation=1.3, gamma=1.1)
        .upscale(factor=4, engine="realesrgan")
    )


def _bw_restore(video: "VideoFile") -> "VideoFile":
    """Black & white footage restoration — denoise, upscale, colorize."""
    return (
        video
        .deinterlace()
        .denoise(engine="nafnet")
        .upscale(factor=4, engine="realesrgan")
        .colorize(engine="deoldify")
        .face_restore(engine="gfpgan")
    )


PRESETS: dict[str, Callable[["VideoFile"], "VideoFile"]] = {
    # FFmpeg only
    "vhs_standard": _vhs_standard,
    "vhs_quality": _vhs_quality,
    "dvd_cleanup": _dvd_cleanup,
    "camcorder": _camcorder,
    "film_8mm": _film_8mm,
    # AI-powered
    "vhs_ai": _vhs_ai,
    "vhs_ai_full": _vhs_ai_full,
    "camcorder_ai": _camcorder_ai,
    "film_8mm_ai": _film_8mm_ai,
    "bw_restore": _bw_restore,
}
