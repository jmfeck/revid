# revid

[![Tests](https://github.com/jmfeck/revid/actions/workflows/tests.yml/badge.svg)](https://github.com/jmfeck/revid/actions/workflows/tests.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/revid)](https://pypi.org/project/revid/)
[![License](https://img.shields.io/pypi/l/revid)](https://github.com/jmfeck/revid/blob/main/LICENSE)

Simple video restoration for Python. Load a video, chain restoration filters, export. That's it.

Built for restoring VHS tapes, old camcorder footage, digitized film, and any degraded video. Uses FFmpeg for fast classic filters, with optional AI-powered models for superior quality.

## Installation

```bash
pip install revid
```

With optional extras:

```bash
pip install revid[ai]   # AI-powered restoration (Real-ESRGAN, GFPGAN, Demucs, etc.)
pip install revid[all]  # everything
```

Requires [FFmpeg](https://ffmpeg.org/) installed and available in your system PATH.

> **Note on AI features:** The `revid[ai]` extras install PyTorch and several AI models (~2GB+). AI engines process video frame-by-frame and are computationally heavy — a dedicated NVIDIA GPU with CUDA support is strongly recommended. FFmpeg-only features run on any machine with no extra dependencies. AI engines were tested on an NVIDIA RTX 5070 Ti (16GB VRAM) with PyTorch CUDA 12.8.

## Usage

```python
import revid as rv

video = rv.read("family_tape_1994.mp4")

# Chain restoration steps and export
video.deinterlace() \
     .denoise(strength=0.5) \
     .color_correct(saturation=1.2, brightness=0.05) \
     .sharpen() \
     .upscale(factor=2) \
     .to_mp4("restored.mp4")

# Inspect intermediate steps
deinterlaced = video.deinterlace()
deinterlaced.to_mp4("step1_deinterlaced.mp4")

denoised = deinterlaced.denoise()
denoised.to_mp4("step2_denoised.mp4")

# Preview a frame without rendering the full video
video.deinterlace().denoise().preview(at=5.0, output="preview.png")

# Use presets
video.preset("vhs_standard").to_mp4("restored.mp4")
video.preset("vhs_ai").to_mp4("restored_ai.mp4")

# GPU-accelerated encoding (NVIDIA NVENC)
video.preset("vhs_standard").to_mp4("restored.mp4", gpu=True)

# AI-powered upscaling
video.upscale(factor=4, engine="realesrgan").to_mp4("upscaled.mp4")

# AI face restoration
video.upscale(factor=2, engine="realesrgan") \
     .face_restore(engine="gfpgan") \
     .to_mp4("faces_restored.mp4")

# AI audio enhancement
video.audio_denoise(engine="demucs").to_mp4("clean_audio.mp4")
video.audio_separate(engine="demucs", stem="vocals").to_mp4("vocals_only.mp4")

# Video metadata
info = video.info()
print(info)
```

Each transformation returns a new object. The original is never modified.

## Presets

### FFmpeg (fast, no dependencies)

| Preset | Description |
|--------|-------------|
| `vhs_standard` | Fast VHS restoration (deinterlace, denoise, color correct, sharpen, upscale 2x) |
| `vhs_quality` | Higher quality VHS restoration (bwdif, nlmeans, deflicker, normalize, upscale 2x) |
| `dvd_cleanup` | Clean up DVD rips (deblock, deband, light denoise) |
| `camcorder` | Restore old camcorder footage (Hi8, MiniDV) |
| `film_8mm` | Restore digitized 8mm / Super 8 film |

### AI-powered (require `revid[ai]`)

| Preset | Description |
|--------|-------------|
| `vhs_ai` | VHS restoration + Real-ESRGAN 4x + GFPGAN face restore |
| `vhs_ai_full` | Maximum quality: NAFNet denoise + Real-ESRGAN 4x + GFPGAN + RIFE 60fps |
| `camcorder_ai` | Camcorder restoration + Real-ESRGAN 4x + GFPGAN |
| `film_8mm_ai` | 8mm restoration + scratch removal + NAFNet + Real-ESRGAN 4x |
| `bw_restore` | Black & white restoration + colorization + face restore |

## Engine Pattern

Every filter method defaults to FFmpeg. Use the `engine` parameter to switch to an AI model:

```python
# FFmpeg (default, fast)
video.upscale(factor=2)
video.upscale(factor=2, engine="ffmpeg", algorithm="lanczos")

# AI (slower, higher quality)
video.upscale(factor=4, engine="realesrgan")

# Same pattern for all methods
video.denoise(strength=0.5)                      # FFmpeg hqdn3d
video.denoise(strength=0.5, algorithm="nlmeans")  # FFmpeg nlmeans
video.denoise(engine="nafnet")                    # AI

video.stabilize()                                 # FFmpeg vidstab
video.stabilize(engine="raft")                    # AI optical flow

video.face_restore(engine="gfpgan")               # AI only
video.colorize(engine="deoldify")                  # AI only
```

## Features

### Video Filters (FFmpeg)

| Feature | Description |
|---------|-------------|
| Deinterlace | Remove combing artifacts (yadif, bwdif, estdif) |
| Denoise | Spatial and temporal denoising (hqdn3d, nlmeans) |
| Sharpen | Unsharp mask and contrast adaptive sharpening (unsharp, cas) |
| Upscale | Resolution upscaling (lanczos, bicubic, spline) |
| Color correct | Brightness, contrast, saturation, gamma (eq) |
| Color curves | Fine-grained tonal adjustments (curves) |
| White balance | Fix color temperature drift (colortemperature) |
| Color levels | Per-channel adjustment (colorlevels) |
| Color normalize | Auto-stretch color range (normalize) |
| Chroma fix | Fix incorrect color space metadata (colorspace) |
| Deflicker | Remove brightness fluctuations (deflicker) |
| Stabilize | 2-pass video stabilization (vidstab) |
| Crop / Auto crop | Remove borders and head switching artifacts |
| Pad | Add borders for aspect ratio correction |
| Rotate / Flip | Fix orientation |
| Deblock | Remove compression artifacts (deblock) |
| Deband | Remove color banding (deband) |
| Decimate | Remove duplicate frames |
| FPS convert | Change framerate |
| Speed adjust | Speed correction for PAL/NTSC mismatch |
| Trim | Cut segments |
| Field order fix | Fix TFF/BFF field order |
| Inverse telecine | Restore 24fps from 3:2 pulldown |
| Concat | Join multiple tapes or segments |
| Grayscale / Sepia | Visual effects |
| Fade in / out | Video and audio fades |
| Reverse | Reverse video and audio |
| Overlay | Watermark or logo overlay |
| Subtitles | Burn subtitles into video |
| Draw text | Burn text onto video |
| Preview | Extract single frame at any point |
| Histogram / Scopes | Visual analysis (histogram, waveform, vectorscope) |

### Audio Filters (FFmpeg)

| Feature | Description |
|---------|-------------|
| Audio denoise | FFT-based noise removal (afftdn) |
| Hum removal | Remove 50/60Hz electrical hum + harmonics |
| Hiss removal | Band-pass filtering for tape hiss |
| Audio normalize | Volume normalization, EBU R128 (loudnorm) |
| Audio equalizer | Bass and treble adjustment |
| Stereo fix | Fix mono/stereo channel issues |
| Audio tempo | Fix audio speed drift |
| Wow & flutter fix | Pitch wobble correction (experimental) |
| Audio fade in / out | Audio fades |

### AI Engines (optional)

| Category | Engines | Description |
|----------|---------|-------------|
| Upscale | Real-ESRGAN, SwinIR, ESPCN, EDSR, BasicVSR++ | Image and video super resolution |
| Denoise | NAFNet, SCUNet, Restormer | Learned denoising for real-world noise |
| Deblur | NAFNet, MPRNet, HINet | Motion and focus blur removal |
| Face restore | GFPGAN, CodeFormer, RestoreFormer | Facial detail recovery |
| Interpolate | RIFE, IFRNet, AMT, FILM | Frame interpolation (30fps to 60fps) |
| Colorize | DeOldify, DDColor, BigColor | Add color to B&W footage |
| Stabilize | RAFT, FlowFormer | Deep optical flow stabilization |
| Inpaint | LaMa, MAT | Fill damaged regions |
| Object remove | ProPainter, E2FGVI | Remove watermarks and overlays |
| Scratch remove | RTN, Old Photo Restore | Tape damage and scratch removal |
| Audio denoise | Demucs, Silero, RNNoise | AI speech enhancement |
| Audio separate | Demucs, Open-Unmix | Source separation (vocals, music, noise) |
| Audio upscale | AudioSR, AERO | Audio bandwidth upscaling |
| Scene detect | PySceneDetect, TransNetV2 | Automatic scene boundary detection |

### Utility

| Feature | Description |
|---------|-------------|
| `rv.read()` | Read any video format |
| `rv.read_mp4()`, `rv.read_avi()`, ... | Format-specific readers |
| `.info()` | Video metadata (codec, resolution, fps, audio) |
| `.preview()` | Extract single frame |
| `.extract_audio()` | Extract audio track |
| `.mute()` | Remove audio |
| `.concat()` | Join multiple videos |
| `.to_mp4()`, `.to_avi()`, ... | Export to format |
| `.render()` | Render with custom options |

## Supported Formats

| Type | Formats |
|------|---------|
| Video | mp4, avi, mov, mkv, webm, flv, ogv, wmv, 3gp, ts, mpeg, mpg |

## License

BSD 3-Clause
