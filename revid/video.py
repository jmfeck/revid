# -*- coding: utf-8 -*-

import os
import subprocess

from revid.base import BaseVideo

# FFmpeg algorithm maps (reused across methods)
_SCALE_ALGORITHMS = {"lanczos": "lanczos", "bicubic": "bicubic", "spline": "spline16"}
_DENOISE_ALGORITHMS = {"hqdn3d", "nlmeans"}
_DEINTERLACE_ALGORITHMS = {"yadif", "bwdif", "estdif"}
_SHARPEN_ALGORITHMS = {"unsharp", "cas"}
_STABILIZE_ALGORITHMS = {"vidstab", "deshake"}

# AI engines per operation type
_UPSCALE_ENGINES = {"realesrgan", "swinir", "espcn", "edsr", "basicvsr"}
_DENOISE_ENGINES = {"nafnet", "scunet", "restormer"}
_STABILIZE_ENGINES = {"raft", "flowformer"}


class VideoFile(BaseVideo):
    """Video file with restoration filters.

    Each filter method returns a new VideoFile with the filter appended.
    Nothing is executed until you call a .to_*() method or .render().
    """

    # =========================================================================
    # Video filters (FFmpeg + AI)
    # =========================================================================

    # --- Deinterlace ---------------------------------------------------------

    def deinterlace(self, engine: str = "ffmpeg", algorithm: str = "yadif", **kwargs) -> "VideoFile":
        """Remove combing artifacts from interlaced video.

        Args:
            engine: "ffmpeg" (default).
            algorithm: "yadif" (default), "bwdif", or "estdif". Only for engine="ffmpeg".
        """
        if engine == "ffmpeg":
            if algorithm == "yadif":
                parity = kwargs.get("parity", -1)
                deint = kwargs.get("deint", 0)
                return self._add_video_filter(f"yadif={deint}:{parity}")
            elif algorithm == "bwdif":
                return self._add_video_filter("bwdif")
            elif algorithm == "estdif":
                return self._add_video_filter("estdif")
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {', '.join(_DEINTERLACE_ALGORITHMS)}")
        raise ValueError(f"Unknown engine: {engine}")

    def field_order(self, order: str = "tff") -> "VideoFile":
        """Fix field order (tff = top field first, bff = bottom field first)."""
        return self._add_video_filter(f"fieldorder={order}")

    def inverse_telecine(self) -> "VideoFile":
        """Restore 24fps from 3:2 pulldown (NTSC telecine)."""
        clone = self._add_video_filter("fieldmatch")
        return clone._add_video_filter("decimate")

    # --- Denoise -------------------------------------------------------------

    def denoise(
        self, strength: float = 0.5, engine: str = "ffmpeg", algorithm: str = "hqdn3d", **kwargs
    ) -> "VideoFile":
        """Reduce noise.

        Args:
            strength: 0.0 to 1.0 (mapped to filter parameters). Only for engine="ffmpeg".
            engine: "ffmpeg" (default), "nafnet", "scunet", "restormer".
            algorithm: "hqdn3d" (default) or "nlmeans". Only for engine="ffmpeg".
        """
        if engine == "ffmpeg":
            if algorithm == "hqdn3d":
                luma = 4.0 + strength * 12.0
                chroma = 3.0 + strength * 9.0
                return self._add_video_filter(f"hqdn3d={luma:.1f}:{chroma:.1f}")
            elif algorithm == "nlmeans":
                s = 3.0 + strength * 12.0
                return self._add_video_filter(f"nlmeans=s={s:.1f}")
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {', '.join(_DENOISE_ALGORITHMS)}")

        if engine in _DENOISE_ENGINES:
            return self._add_ai_step({"type": "denoise", "engine": engine, "strength": strength, **kwargs})

        raise ValueError(f"Unknown engine: {engine}. Available: ffmpeg, {', '.join(_DENOISE_ENGINES)}")

    def temporal_denoise(self, strength: float = 0.5) -> "VideoFile":
        """Reduce noise using neighboring frames (FFmpeg hqdn3d temporal)."""
        luma = 4.0 + strength * 12.0
        tmp = 4.0 + strength * 12.0
        return self._add_video_filter(f"hqdn3d={luma:.1f}:{luma:.1f}:{tmp:.1f}:{tmp:.1f}")

    # --- Sharpen -------------------------------------------------------------

    def sharpen(self, amount: float = 1.0, engine: str = "ffmpeg", algorithm: str = "unsharp") -> "VideoFile":
        """Sharpen the video.

        Args:
            amount: Sharpening strength (0.0 to 3.0). Only for engine="ffmpeg".
            engine: "ffmpeg" (default).
            algorithm: "unsharp" (default) or "cas" (contrast adaptive). Only for engine="ffmpeg".
        """
        if engine == "ffmpeg":
            if algorithm == "unsharp":
                return self._add_video_filter(f"unsharp=5:5:{amount:.2f}:5:5:{amount * 0.5:.2f}")
            elif algorithm == "cas":
                s = min(amount / 3.0, 1.0)
                return self._add_video_filter(f"cas={s:.2f}")
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {', '.join(_SHARPEN_ALGORITHMS)}")
        raise ValueError(f"Unknown engine: {engine}")

    # --- Upscale -------------------------------------------------------------

    def upscale(self, factor: int = 2, engine: str = "ffmpeg", algorithm: str = "lanczos", **kwargs) -> "VideoFile":
        """Upscale video resolution.

        Args:
            factor: Scale multiplier (2, 3, 4).
            engine: "ffmpeg" (default), "realesrgan", "swinir", "espcn", "edsr", "basicvsr".
            algorithm: "lanczos" (default), "bicubic", or "spline". Only for engine="ffmpeg".
            model: Model variant (engine-specific). E.g. "realesrgan-x4plus" for realesrgan.
        """
        if engine == "ffmpeg":
            flags = _SCALE_ALGORITHMS.get(algorithm, "lanczos")
            return self._add_video_filter(f"scale=iw*{factor}:ih*{factor}:flags={flags}")

        if engine in _UPSCALE_ENGINES:
            return self._add_ai_step(
                {
                    "type": "upscale",
                    "engine": engine,
                    "factor": factor,
                    **kwargs,
                }
            )

        raise ValueError(f"Unknown engine: {engine}. Available: ffmpeg, {', '.join(_UPSCALE_ENGINES)}")

    def resize(self, width: int, height: int, algorithm: str = "lanczos") -> "VideoFile":
        """Resize to exact dimensions.

        When used after AI steps, the resize is applied during re-encoding
        (after AI processing), not in the intermediate FFmpeg pass.
        """
        flags = _SCALE_ALGORITHMS.get(algorithm, "lanczos")
        filter_str = f"scale={width}:{height}:flags={flags}"

        # If there are AI steps, this resize should happen AFTER AI processing
        if self._ai_steps:
            clone = self._clone()
            clone._post_ai_filters.append(filter_str)
            return clone

        return self._add_video_filter(filter_str)

    # --- Color ---------------------------------------------------------------

    def color_correct(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
    ) -> "VideoFile":
        """Adjust color properties.

        Args:
            brightness: -1.0 to 1.0 (0.0 = no change).
            contrast: 0.0 to 3.0 (1.0 = no change).
            saturation: 0.0 to 3.0 (1.0 = no change).
            gamma: 0.1 to 10.0 (1.0 = no change).
        """
        return self._add_video_filter(
            f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"
        )

    def white_balance(self, temperature: int = 6500) -> "VideoFile":
        """Fix color temperature drift.

        Args:
            temperature: Color temperature in Kelvin (default 6500 = daylight).
        """
        return self._add_video_filter(f"colortemperature=temperature={temperature}")

    def color_levels(
        self,
        rimin: float = 0,
        rimax: float = 1,
        gimin: float = 0,
        gimax: float = 1,
        bimin: float = 0,
        bimax: float = 1,
    ) -> "VideoFile":
        """Adjust per-channel color levels."""
        return self._add_video_filter(
            f"colorlevels=rimin={rimin}:rimax={rimax}:gimin={gimin}:gimax={gimax}:bimin={bimin}:bimax={bimax}"
        )

    def color_normalize(self) -> "VideoFile":
        """Auto-stretch color range."""
        return self._add_video_filter("normalize")

    def curves(self, preset: str = "none") -> "VideoFile":
        """Apply color curves.

        Args:
            preset: "none", "color_negative", "cross_process", "darker",
                    "increase_contrast", "lighter", "linear_contrast",
                    "medium_contrast", "negative", "strong_contrast", "vintage".
        """
        return self._add_video_filter(f"curves=preset={preset}")

    def chroma_fix(self, src: str = "bt601-6-525", dst: str = "bt709") -> "VideoFile":
        """Fix incorrect color space metadata.

        Args:
            src: Source color space (default "bt601-6-525" for NTSC VHS).
                 Options: "bt470m", "bt470bg", "bt601-6-525", "bt601-6-625", "bt709", "bt2020".
            dst: Destination color space (default "bt709").
        """
        return self._add_video_filter(f"colorspace=all={dst}:iall={src}")

    # --- Stabilize -----------------------------------------------------------

    def stabilize(
        self,
        engine: str = "ffmpeg",
        algorithm: str = "vidstab",
        shakiness: int = 5,
        accuracy: int = 15,
        smoothing: int = 10,
        **kwargs,
    ) -> "VideoFile":
        """Stabilize shaky video.

        Args:
            engine: "ffmpeg" (default), "raft", "flowformer".
            algorithm: "vidstab" (2-pass, default) or "deshake" (single-pass). Only for engine="ffmpeg".
            shakiness: 1-10 (default 5). Only for vidstab.
            accuracy: 1-15 (default 15). Only for vidstab.
            smoothing: Frames for smoothing (default 10). Only for vidstab.
        """
        if engine == "ffmpeg":
            if algorithm == "deshake":
                rx = kwargs.get("rx", 16)
                ry = kwargs.get("ry", 16)
                return self._add_video_filter(f"deshake=rx={rx}:ry={ry}")
            elif algorithm == "vidstab":
                clone = self._clone()
                clone._stabilize_params = {
                    "shakiness": shakiness,
                    "accuracy": accuracy,
                    "smoothing": smoothing,
                }
                return clone
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {', '.join(_STABILIZE_ALGORITHMS)}")

        if engine in _STABILIZE_ENGINES:
            return self._add_ai_step({"type": "stabilize", "engine": engine, **kwargs})

        raise ValueError(f"Unknown engine: {engine}. Available: ffmpeg, {', '.join(_STABILIZE_ENGINES)}")

    # --- Crop / Geometry -----------------------------------------------------

    def crop(self, x: int, y: int, width: int, height: int) -> "VideoFile":
        """Crop the video."""
        return self._add_video_filter(f"crop={width}:{height}:{x}:{y}")

    def auto_crop(self) -> "VideoFile":
        """Detect and remove black borders."""
        return self._add_video_filter("cropdetect")

    def pad(self, width: int, height: int, x: int = -1, y: int = -1, color: str = "black") -> "VideoFile":
        """Add padding/borders."""
        return self._add_video_filter(f"pad={width}:{height}:{x}:{y}:color={color}")

    def aspect_ratio(self, ratio: str = "16/9") -> "VideoFile":
        """Set display aspect ratio."""
        return self._add_video_filter(f"setdar={ratio}")

    def rotate(self, angle: int = 90) -> "VideoFile":
        """Rotate the video (90, 180, or 270 degrees)."""
        if angle == 90:
            return self._add_video_filter("transpose=1")
        elif angle == 180:
            clone = self._add_video_filter("hflip")
            return clone._add_video_filter("vflip")
        elif angle == 270:
            return self._add_video_filter("transpose=2")
        raise ValueError("Angle must be 90, 180, or 270")

    def flip_horizontal(self) -> "VideoFile":
        """Flip video horizontally."""
        return self._add_video_filter("hflip")

    def flip_vertical(self) -> "VideoFile":
        """Flip video vertically."""
        return self._add_video_filter("vflip")

    def vignette_remove(self, amount: float = 0.5) -> "VideoFile":
        """Correct dark corners (vignetting)."""
        angle = f"PI/{2.0 / amount:.1f}"
        return self._add_video_filter(f"vignette=angle={angle}:mode=backward")

    def lens_correction(self, k1: float = 0.0, k2: float = 0.0) -> "VideoFile":
        """Fix lens distortion / chromatic aberration."""
        return self._add_video_filter(f"lenscorrection=k1={k1}:k2={k2}")

    # --- Temporal / FPS ------------------------------------------------------

    def deflicker(self, size: int = 5, mode: str = "am") -> "VideoFile":
        """Remove brightness fluctuations.

        Args:
            size: Window size in frames (default 5).
            mode: "am" (arithmetic mean) or "gm" (geometric mean).
        """
        return self._add_video_filter(f"deflicker=size={size}:mode={mode}")

    def decimate(self) -> "VideoFile":
        """Remove duplicate frames."""
        return self._add_video_filter("mpdecimate")

    def set_fps(self, rate: float = 30.0) -> "VideoFile":
        """Change framerate."""
        return self._add_video_filter(f"fps={rate}")

    def speed(self, factor: float = 1.0) -> "VideoFile":
        """Adjust playback speed (video + audio)."""
        clone = self._add_video_filter(f"setpts={1.0 / factor:.4f}*PTS")
        if factor != 1.0:
            clone = clone._add_audio_filter(f"atempo={factor}")
        return clone

    def trim(self, start: float = 0.0, end: float | None = None) -> "VideoFile":
        """Cut a segment from the video."""
        clone = self._clone()
        clone._pre_args.extend(["-ss", str(start)])
        if end is not None:
            clone._pre_args.extend(["-to", str(end)])
        return clone

    # --- Effects / Visual ----------------------------------------------------

    def grayscale(self) -> "VideoFile":
        """Convert video to grayscale."""
        return self._add_video_filter("format=gray")

    def sepia(self) -> "VideoFile":
        """Apply sepia tone effect."""
        return self._add_video_filter("colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131")

    def fade_in(self, duration: float = 2.0) -> "VideoFile":
        """Add a fade-in from black at the start.

        Args:
            duration: Fade duration in seconds (default 2.0).
        """
        return self._add_video_filter(f"fade=t=in:st=0:d={duration}")

    def fade_out(self, duration: float = 2.0, start: float | None = None) -> "VideoFile":
        """Add a fade-out to black.

        Args:
            duration: Fade duration in seconds (default 2.0).
            start: Start time in seconds. If None, calculated from video duration.
        """
        if start is not None:
            return self._add_video_filter(f"fade=t=out:st={start}:d={duration}")
        d = self.duration
        if d is None:
            raise ValueError("Cannot determine video duration. Provide a start time.")
        return self._add_video_filter(f"fade=t=out:st={d - duration}:d={duration}")

    def audio_fade_in(self, duration: float = 2.0) -> "VideoFile":
        """Add an audio fade-in."""
        return self._add_audio_filter(f"afade=t=in:st=0:d={duration}")

    def audio_fade_out(self, duration: float = 2.0, start: float | None = None) -> "VideoFile":
        """Add an audio fade-out."""
        if start is not None:
            return self._add_audio_filter(f"afade=t=out:st={start}:d={duration}")
        d = self.duration
        if d is None:
            raise ValueError("Cannot determine video duration. Provide a start time.")
        return self._add_audio_filter(f"afade=t=out:st={d - duration}:d={duration}")

    def reverse(self) -> "VideoFile":
        """Reverse the video (and audio)."""
        clone = self._add_video_filter("reverse")
        return clone._add_audio_filter("areverse")

    def smart_blur(
        self,
        luma_radius: float = 1.0,
        luma_strength: float = 1.0,
        chroma_radius: float = -1.0,
        chroma_strength: float = -1.0,
    ) -> "VideoFile":
        """Selective blur for luma or chroma channels.

        Args:
            luma_radius: Luma blur radius (default 1.0).
            luma_strength: Luma blur strength (default 1.0).
            chroma_radius: Chroma blur radius (-1 = same as luma).
            chroma_strength: Chroma blur strength (-1 = same as luma).
        """
        if chroma_radius < 0:
            chroma_radius = luma_radius
        if chroma_strength < 0:
            chroma_strength = luma_strength
        return self._add_video_filter(
            f"smartblur=lr={luma_radius}:ls={luma_strength}:cr={chroma_radius}:cs={chroma_strength}"
        )

    # --- Overlay / Text ------------------------------------------------------

    def overlay(self, image_path: str, x: int = 0, y: int = 0) -> "VideoFile":
        """Overlay an image (watermark, logo) on the video.

        Args:
            image_path: Path to the overlay image.
            x: Horizontal position (default 0).
            y: Vertical position (default 0).
        """
        clone = self._clone()
        clone._extra_inputs.append(os.path.abspath(image_path))
        clone._complex_filter = f"overlay={x}:{y}"
        return clone

    def subtitles(self, subtitle_path: str, force_style: str | None = None) -> "VideoFile":
        """Burn subtitles into the video.

        Args:
            subtitle_path: Path to subtitle file (.srt, .ass, .ssa).
            force_style: Optional ASS style override string.
        """
        path = os.path.abspath(subtitle_path).replace("\\", "/").replace(":", "\\\\:")
        if force_style:
            return self._add_video_filter(f"subtitles='{path}':force_style='{force_style}'")
        return self._add_video_filter(f"subtitles='{path}'")

    def drawtext(
        self, text: str, x: str = "10", y: str = "10", fontsize: int = 24, fontcolor: str = "white"
    ) -> "VideoFile":
        """Burn text onto the video.

        Args:
            text: Text to display.
            x: Horizontal position (default "10"). Supports "(w-text_w)/2" for center.
            y: Vertical position (default "10"). Supports "(h-text_h)/2" for center.
            fontsize: Font size (default 24).
            fontcolor: Font color (default "white").
        """
        escaped = text.replace("'", "\\'").replace(":", "\\:")
        return self._add_video_filter(
            f"drawtext=text='{escaped}':x={x}:y={y}:fontsize={fontsize}:fontcolor={fontcolor}"
        )

    # --- Analysis / Scopes ---------------------------------------------------

    def histogram(self, output: str | None = None, at: float = 0.0) -> str:
        """Generate a color histogram frame."""
        if output is None:
            name = os.path.splitext(os.path.basename(self._path))[0]
            output = os.path.join(os.path.dirname(self._path), f"{name}_histogram.png")
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        filters = list(self._video_filters) + ["histogram"]
        cmd = ["ffmpeg", "-y", "-ss", str(at), "-i", self._path, "-frames:v", "1", "-vf", ",".join(filters), output]
        subprocess.run(cmd, check=True)
        return output

    def waveform(self, output: str | None = None, at: float = 0.0) -> str:
        """Generate a waveform scope frame."""
        if output is None:
            name = os.path.splitext(os.path.basename(self._path))[0]
            output = os.path.join(os.path.dirname(self._path), f"{name}_waveform.png")
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        filters = list(self._video_filters) + ["waveform"]
        cmd = ["ffmpeg", "-y", "-ss", str(at), "-i", self._path, "-frames:v", "1", "-vf", ",".join(filters), output]
        subprocess.run(cmd, check=True)
        return output

    def vectorscope(self, output: str | None = None, at: float = 0.0) -> str:
        """Generate a vectorscope frame (color distribution)."""
        if output is None:
            name = os.path.splitext(os.path.basename(self._path))[0]
            output = os.path.join(os.path.dirname(self._path), f"{name}_vectorscope.png")
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        filters = list(self._video_filters) + ["vectorscope=mode=color2"]
        cmd = ["ffmpeg", "-y", "-ss", str(at), "-i", self._path, "-frames:v", "1", "-vf", ",".join(filters), output]
        subprocess.run(cmd, check=True)
        return output

    # --- Artifact removal ----------------------------------------------------

    def deblock(self, strength: float = 0.5) -> "VideoFile":
        """Remove blocky compression artifacts."""
        s = int(strength * 15)
        return self._add_video_filter(f"deblock=filter=strong:block={s}")

    def deband(self) -> "VideoFile":
        """Remove color banding."""
        return self._add_video_filter("deband")

    # =========================================================================
    # AI-only features (no FFmpeg equivalent)
    # =========================================================================

    def face_restore(self, engine: str = "gfpgan", **kwargs) -> "VideoFile":
        """Restore faces in the video using AI.

        Args:
            engine: "gfpgan" (default), "codeformer", "restoreformer".
            fidelity: 0.0-1.0, balance between quality and fidelity (CodeFormer only).
            upscale: Upscale factor for restored faces (default 2).
        """
        return self._add_ai_step({"type": "face_restore", "engine": engine, **kwargs})

    def interpolate(self, target_fps: float = 60.0, engine: str = "rife", **kwargs) -> "VideoFile":
        """Generate intermediate frames using AI (frame interpolation).

        Args:
            target_fps: Target framerate (default 60).
            engine: "rife" (default), "ifrnet", "amt", "film".
            multiplier: Frame multiplier (default auto-calculated from target_fps).
        """
        current_fps = self.fps or 30.0
        multiplier = kwargs.pop("multiplier", max(1, round(target_fps / current_fps)))
        return self._add_ai_step(
            {
                "type": "interpolate",
                "engine": engine,
                "target_fps": target_fps,
                "multiplier": multiplier,
                **kwargs,
            }
        )

    def colorize(self, engine: str = "deoldify", **kwargs) -> "VideoFile":
        """Add color to black and white footage using AI.

        Args:
            engine: "deoldify" (default), "ddcolor", "bigcolor".
        """
        return self._add_ai_step({"type": "colorize", "engine": engine, **kwargs})

    def deblur(self, engine: str = "nafnet", **kwargs) -> "VideoFile":
        """Remove motion blur or focus blur using AI.

        Args:
            engine: "nafnet" (default), "mprnet", "hinet".
        """
        return self._add_ai_step({"type": "deblur", "engine": engine, **kwargs})

    def inpaint(self, mask_path: str, engine: str = "lama", **kwargs) -> "VideoFile":
        """Fill in damaged or corrupted regions using AI.

        Args:
            mask_path: Path to mask image (white = regions to fill).
            engine: "lama" (default), "mat".
        """
        return self._add_ai_step(
            {
                "type": "inpaint",
                "engine": engine,
                "mask_path": os.path.abspath(mask_path),
                **kwargs,
            }
        )

    def object_remove(self, mask_path: str, engine: str = "propainter", **kwargs) -> "VideoFile":
        """Remove objects (watermarks, timestamps) using AI.

        Args:
            mask_path: Path to mask image (white = regions to remove).
            engine: "propainter" (default), "e2fgvi".
        """
        return self._add_ai_step(
            {
                "type": "object_remove",
                "engine": engine,
                "mask_path": os.path.abspath(mask_path),
                **kwargs,
            }
        )

    def scratch_remove(self, engine: str = "rtn", **kwargs) -> "VideoFile":
        """Detect and remove tape damage, scratches, dropouts using AI.

        Args:
            engine: "rtn" (default), "old_photo_restore".
        """
        return self._add_ai_step({"type": "scratch_remove", "engine": engine, **kwargs})

    def scene_detect(self, engine: str = "pyscenedetect", **kwargs) -> "VideoFile":
        """Detect scene boundaries.

        Args:
            engine: "pyscenedetect" (default), "transnetv2".
        """
        return self._add_ai_step({"type": "scene_detect", "engine": engine, **kwargs})

    # =========================================================================
    # Audio filters
    # =========================================================================

    def audio_denoise(
        self, engine: str = "ffmpeg", noise_reduction: float = 12.0, noise_floor: float = -30.0, **kwargs
    ) -> "VideoFile":
        """Remove audio noise.

        Args:
            engine: "ffmpeg" (default), "demucs", "silero", "rnnoise".
            noise_reduction: dB of noise reduction (default 12). Only for engine="ffmpeg".
            noise_floor: Noise floor in dB (default -30). Only for engine="ffmpeg".
        """
        if engine == "ffmpeg":
            return self._add_audio_filter(f"afftdn=nr={noise_reduction}:nf={noise_floor}")
        return self._add_ai_step({"type": "audio_denoise", "engine": engine, **kwargs})

    def audio_separate(self, engine: str = "demucs", **kwargs) -> "VideoFile":
        """Separate voice from music and noise using AI.

        Args:
            engine: "demucs" (default), "open_unmix".
        """
        return self._add_ai_step({"type": "audio_separate", "engine": engine, **kwargs})

    def audio_upscale(self, engine: str = "audiosr", **kwargs) -> "VideoFile":
        """Upscale audio bandwidth (make muffled tape audio clearer) using AI.

        Args:
            engine: "audiosr" (default), "aero".
        """
        return self._add_ai_step({"type": "audio_upscale", "engine": engine, **kwargs})

    def hum_remove(self, frequency: float = 60.0, harmonics: int = 1) -> "VideoFile":
        """Remove electrical hum and its harmonics.

        Args:
            frequency: Base hum frequency (default 60Hz). Use 50 for EU/BR.
            harmonics: Number of harmonics to remove (default 1 = only fundamental).
                       Use 5 to remove 60, 120, 180, 240, 300 Hz.
        """
        clone = self._clone()
        for i in range(1, harmonics + 1):
            clone = clone._add_audio_filter(f"bandreject=frequency={frequency * i}:width_type=h:width=10")
        return clone

    def hiss_remove(self, low: int = 200, high: int = 8000) -> "VideoFile":
        """Remove tape hiss with band-pass filtering.

        Args:
            low: Low cutoff frequency in Hz (default 200).
            high: High cutoff frequency in Hz (default 8000).
        """
        clone = self._add_audio_filter(f"highpass=f={low}")
        return clone._add_audio_filter(f"lowpass=f={high}")

    def audio_normalize(self, mode: str = "loudnorm") -> "VideoFile":
        """Normalize audio volume.

        Args:
            mode: "loudnorm" (EBU R128, default) or "dynaudnorm" (dynamic).
        """
        if mode in ("loudnorm", "dynaudnorm"):
            return self._add_audio_filter(mode)
        raise ValueError(f"Unknown mode: {mode}. Available: loudnorm, dynaudnorm")

    def audio_equalizer(self, bass: float = 0.0, treble: float = 0.0) -> "VideoFile":
        """Adjust audio tone.

        Args:
            bass: Bass gain in dB (-20 to 20, default 0).
            treble: Treble gain in dB (-20 to 20, default 0).
        """
        clone = self._clone()
        if bass != 0.0:
            clone = clone._add_audio_filter(f"bass=g={bass}")
        if treble != 0.0:
            clone = clone._add_audio_filter(f"treble=g={treble}")
        return clone

    def audio_stereo_fix(self, layout: str = "stereo") -> "VideoFile":
        """Fix audio channel layout.

        Args:
            layout: "mono" or "stereo" (default).
        """
        if layout == "mono":
            return self._add_audio_filter("pan=mono|c0=0.5*c0+0.5*c1")
        elif layout == "stereo":
            return self._add_audio_filter("pan=stereo|c0=c0|c1=c1")
        raise ValueError(f"Unknown layout: {layout}. Available: mono, stereo")

    def audio_tempo(self, factor: float = 1.0) -> "VideoFile":
        """Fix audio speed drift."""
        return self._add_audio_filter(f"atempo={factor}")

    def wow_flutter_fix(self, smooth: float = 10.0) -> "VideoFile":
        """Fix wow and flutter (pitch wobble from inconsistent tape speed).

        Uses pitch detection and correction to stabilize audio pitch.

        Args:
            smooth: Smoothing window in milliseconds (default 10.0).
                    Higher = more correction, but may sound unnatural.
        """
        return self._add_audio_filter(f"asetrate=48000,aresample=48000,rubberband=pitch=1.0:smoothing={smooth}")

    # =========================================================================
    # Presets
    # =========================================================================

    def preset(self, name: str) -> "VideoFile":
        """Apply a named restoration preset.

        Available presets:
            - "vhs_standard": Fast FFmpeg-only VHS restoration.
            - "vhs_quality": Higher quality VHS restoration.
            - "dvd_cleanup": Clean up DVD rips.
            - "camcorder": Restore old camcorder footage.
            - "film_8mm": Restore digitized 8mm/Super 8 film.
        """
        from revid.presets import PRESETS

        if name not in PRESETS:
            available = ", ".join(sorted(PRESETS.keys()))
            raise ValueError(f"Unknown preset: {name}. Available: {available}")

        return PRESETS[name](self)
