# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-04

### Added

- Initial release
- Core `VideoFile` class with immutable filter chaining
- 40+ FFmpeg video filters (deinterlace, denoise, sharpen, upscale, color correct, stabilize, crop, etc.)
- 10+ FFmpeg audio filters (denoise, hum removal, hiss removal, normalize, equalizer, etc.)
- Engine pattern for swapping between FFmpeg and AI models
- 38 AI engine implementations across 14 categories (upscale, denoise, face restore, colorize, etc.)
- Real-ESRGAN upscaling (PyTorch CUDA)
- GFPGAN face restoration (PyTorch CUDA)
- Demucs audio enhancement and source separation
- 10 restoration presets (5 FFmpeg, 5 AI-powered)
- GPU-accelerated encoding via NVENC
- Preview frame extraction
- Video metadata probing (info, duration, size, fps)
- Multiple output formats (mp4, avi, mov, mkv, webm, flv, ts, mpeg)
- Generic `vr.read()` and format-specific readers (`vr.read_mp4()`, etc.)
- Concat, extract audio, mute utilities
- Histogram, waveform, and vectorscope analysis
