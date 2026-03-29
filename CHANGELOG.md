# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-03-29

### Changed

- Renamed recommended import alias from `vr` to `rv` (`import revid as rv`)
- Added AI hardware requirements note to README

### Fixed

- Guarded numpy/PIL imports in engine modules for environments without AI deps
- Fixed `_try_import` to catch `AttributeError` in engine registry
- Fixed setuptools license classifier conflict
- Fixed ruff formatting across all source files

## [0.1.1] - 2026-03-29

### Fixed

- Fixed PyPI publish pipeline (trusted publisher)

## [0.1.0] - 2026-03-29

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
- Generic `rv.read()` and format-specific readers (`rv.read_mp4()`, etc.)
- Concat, extract audio, mute utilities
- Histogram, waveform, and vectorscope analysis
