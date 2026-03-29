# -*- coding: utf-8 -*-

import json
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy


class BaseVideo:
    """Base class for video files. Manages the FFmpeg filter graph and rendering."""

    SUPPORTED_FORMATS = {"mp4", "avi", "mov", "mkv", "webm", "flv", "ogv", "wmv", "3gp", "ts", "mpeg", "mpg"}

    def __init__(self, path: str):
        self._path = os.path.abspath(path)
        if not os.path.isfile(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")
        self._video_filters: list[str] = []
        self._audio_filters: list[str] = []
        self._pre_args: list[str] = []
        self._post_args: list[str] = []
        self._probe_cache: dict | None = None
        self._stabilize_params: dict | None = None
        self._extra_inputs: list[str] = []
        self._complex_filter: str | None = None
        self._ai_steps: list[dict] = []
        self._post_ai_filters: list[str] = []

    def __repr__(self) -> str:
        ext = os.path.splitext(self._path)[1].lstrip(".")
        name = os.path.basename(self._path)
        n_filters = len(self._video_filters) + len(self._audio_filters)
        return f"<VideoFile '{name}' ({ext}) filters={n_filters}>"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @property
    def path(self) -> str:
        return self._path

    # -------------------------------------------------------------------------
    # Probe (lazy, cached)
    # -------------------------------------------------------------------------

    def _probe(self) -> dict:
        """Run ffprobe and cache the result."""
        if self._probe_cache is not None:
            return self._probe_cache

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            self._path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self._probe_cache = json.loads(result.stdout)
        return self._probe_cache

    def _video_stream(self) -> dict | None:
        """Return the first video stream from probe data."""
        for stream in self._probe().get("streams", []):
            if stream.get("codec_type") == "video":
                return stream
        return None

    def _audio_stream(self) -> dict | None:
        """Return the first audio stream from probe data."""
        for stream in self._probe().get("streams", []):
            if stream.get("codec_type") == "audio":
                return stream
        return None

    @property
    def duration(self) -> float | None:
        fmt = self._probe().get("format", {})
        d = fmt.get("duration")
        return float(d) if d else None

    @property
    def size(self) -> tuple[int, int] | None:
        vs = self._video_stream()
        if vs and "width" in vs and "height" in vs:
            return (int(vs["width"]), int(vs["height"]))
        return None

    @property
    def width(self) -> int | None:
        s = self.size
        return s[0] if s else None

    @property
    def height(self) -> int | None:
        s = self.size
        return s[1] if s else None

    @property
    def fps(self) -> float | None:
        vs = self._video_stream()
        if vs and "r_frame_rate" in vs:
            num, den = vs["r_frame_rate"].split("/")
            return float(num) / float(den) if float(den) != 0 else None
        return None

    # -------------------------------------------------------------------------
    # Filter graph (immutable chaining)
    # -------------------------------------------------------------------------

    def _clone(self) -> "BaseVideo":
        """Return a copy with the same source and filter graph."""
        new = object.__new__(self.__class__)
        new._path = self._path
        new._video_filters = deepcopy(self._video_filters)
        new._audio_filters = deepcopy(self._audio_filters)
        new._pre_args = deepcopy(self._pre_args)
        new._post_args = deepcopy(self._post_args)
        new._probe_cache = self._probe_cache
        new._stabilize_params = deepcopy(self._stabilize_params)
        new._extra_inputs = deepcopy(self._extra_inputs)
        new._complex_filter = self._complex_filter
        new._ai_steps = deepcopy(self._ai_steps)
        new._post_ai_filters = deepcopy(self._post_ai_filters)
        return new

    def _add_video_filter(self, filter_str: str) -> "BaseVideo":
        clone = self._clone()
        clone._video_filters.append(filter_str)
        return clone

    def _add_audio_filter(self, filter_str: str) -> "BaseVideo":
        clone = self._clone()
        clone._audio_filters.append(filter_str)
        return clone

    def _add_ai_step(self, step: dict) -> "BaseVideo":
        """Register an AI processing step to be executed during render.

        Each step is a dict with:
            - "type": the operation (e.g., "upscale", "denoise", "face_restore")
            - "engine": the AI engine name (e.g., "realesrgan", "nafnet")
            - plus engine-specific parameters
        """
        clone = self._clone()
        clone._ai_steps.append(step)
        return clone

    # -------------------------------------------------------------------------
    # Output path helper
    # -------------------------------------------------------------------------

    def _output_path(self, output: str | None, default_suffix: str = "_restored", fmt: str = "mp4") -> str:
        if output:
            return os.path.abspath(output)
        name = os.path.splitext(os.path.basename(self._path))[0]
        directory = os.path.dirname(self._path)
        return os.path.join(directory, f"{name}{default_suffix}.{fmt}")

    # -------------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------------

    def _build_command(self, output_path: str, gpu: bool = False) -> list[str]:
        """Build the full FFmpeg command."""
        cmd = ["ffmpeg", "-y"]
        cmd.extend(self._pre_args)
        cmd.extend(["-i", self._path])

        for extra in self._extra_inputs:
            cmd.extend(["-i", extra])

        if self._complex_filter:
            vf = ",".join(self._video_filters) if self._video_filters else ""
            if vf:
                cmd.extend(["-filter_complex", f"[0:v]{vf}[base];[base][1:v]{self._complex_filter}"])
            else:
                cmd.extend(["-filter_complex", f"[0:v][1:v]{self._complex_filter}"])
        elif self._video_filters:
            cmd.extend(["-vf", ",".join(self._video_filters)])

        if self._audio_filters:
            cmd.extend(["-af", ",".join(self._audio_filters)])

        if not self._audio_filters:
            cmd.extend(["-c:a", "copy"])

        if gpu:
            ext = os.path.splitext(output_path)[1].lstrip(".")
            encoder = self._gpu_encoder(ext)
            if encoder:
                cmd.extend(["-c:v", encoder, "-preset", "p4"])

        cmd.extend(self._post_args)
        cmd.append(output_path)
        return cmd

    # -------------------------------------------------------------------------
    # GPU helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _gpu_encoder(fmt: str) -> str | None:
        """Return NVENC encoder name for the given format, or None."""
        nvenc_map = {
            "mp4": "h264_nvenc",
            "mkv": "h264_nvenc",
            "mov": "h264_nvenc",
            "avi": "h264_nvenc",
            "ts": "h264_nvenc",
            "flv": "h264_nvenc",
        }
        return nvenc_map.get(fmt)

    def render(self, output: str | None = None, fmt: str = "mp4", gpu: bool = False) -> str:
        """Execute the filter graph and write the output file.

        Args:
            output: Output file path.
            fmt: Output format (default "mp4").
            gpu: Use NVENC GPU encoding (default False).
        """
        output_path = self._output_path(output, fmt=fmt)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if self._stabilize_params:
            self._render_stabilized(output_path, gpu=gpu)
            return output_path

        if self._ai_steps:
            self._render_with_ai(output_path, gpu=gpu)
            return output_path

        cmd = self._build_command(output_path, gpu=gpu)
        subprocess.run(cmd, check=True)
        return output_path

    def _render_with_ai(self, output_path: str, gpu: bool = False) -> None:
        """Render pipeline that includes AI processing steps.

        Flow:
            1. Apply FFmpeg video/audio filters → intermediate video
            2. Split AI steps into video steps and audio steps
            3. Extract frames, run video AI steps
            4. Extract audio, run audio AI steps
            5. Re-encode processed frames + processed audio → output
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Apply FFmpeg filters to an intermediate file
            intermediate = os.path.join(tmpdir, "intermediate.mp4")
            if self._video_filters or self._audio_filters:
                ffmpeg_clone = self._clone()
                ffmpeg_clone._ai_steps = []
                ffmpeg_clone._stabilize_params = None
                cmd = ffmpeg_clone._build_command(intermediate)
                subprocess.run(cmd, check=True)
                source = intermediate
            else:
                source = self._path

            # Step 2: Split AI steps into video and audio
            video_ai_steps = [s for s in self._ai_steps if not s["type"].startswith("audio_")]
            audio_ai_steps = [s for s in self._ai_steps if s["type"].startswith("audio_")]

            # Step 3: Process video AI steps (frames)
            frames_dir = None
            if video_ai_steps:
                frames_in = os.path.join(tmpdir, "frames_in")
                os.makedirs(frames_in)
                extract_cmd = ["ffmpeg", "-y", "-i", source, f"{frames_in}/%08d.png"]
                subprocess.run(extract_cmd, check=True)

                frames_dir = frames_in
                for i, step in enumerate(video_ai_steps):
                    frames_out = os.path.join(tmpdir, f"frames_ai_{i}")
                    os.makedirs(frames_out)
                    self._run_ai_step(step, frames_dir, frames_out)
                    frames_dir = frames_out

            # Step 4: Process audio AI steps
            audio_path = None
            if audio_ai_steps:
                # Extract audio from source
                audio_in_dir = os.path.join(tmpdir, "audio_in")
                os.makedirs(audio_in_dir)
                audio_in = os.path.join(audio_in_dir, "audio.wav")
                subprocess.run(["ffmpeg", "-y", "-i", source, "-vn", audio_in], check=True)

                audio_dir = audio_in_dir
                for i, step in enumerate(audio_ai_steps):
                    audio_out_dir = os.path.join(tmpdir, f"audio_ai_{i}")
                    os.makedirs(audio_out_dir)
                    self._run_ai_step(step, audio_dir, audio_out_dir)
                    audio_dir = audio_out_dir

                # Find the processed audio file
                import glob as _glob
                audio_files = _glob.glob(os.path.join(audio_dir, "audio.*"))
                if audio_files:
                    audio_path = audio_files[0]

            # Step 5: Re-encode
            # Get FPS from original file
            probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                         "-show_streams", self._path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_result.stdout)
            fps = "30"
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video" and "r_frame_rate" in stream:
                    fps = stream["r_frame_rate"]
                    break

            if frames_dir:
                # Video was processed — re-encode from frames
                encode_cmd = ["ffmpeg", "-y", "-framerate", fps,
                              "-i", f"{frames_dir}/%08d.png"]

                if audio_path:
                    encode_cmd.extend(["-i", audio_path])
                    encode_cmd.extend(["-map", "0:v", "-map", "1:a"])
                else:
                    encode_cmd.extend(["-i", source])
                    encode_cmd.extend(["-map", "0:v", "-map", "1:a?", "-c:a", "copy"])

                # Apply post-AI filters (e.g. resize after upscale)
                if self._post_ai_filters:
                    encode_cmd.extend(["-vf", ",".join(self._post_ai_filters)])

                encode_cmd.extend(["-pix_fmt", "yuv420p"])
            else:
                # Only audio was processed — copy video, replace audio
                encode_cmd = ["ffmpeg", "-y", "-i", source]
                if audio_path:
                    encode_cmd.extend(["-i", audio_path])
                    encode_cmd.extend(["-map", "0:v", "-map", "1:a", "-c:v", "copy"])
                else:
                    encode_cmd.extend(["-c:v", "copy", "-c:a", "copy"])

            if gpu and frames_dir:
                ext = os.path.splitext(output_path)[1].lstrip(".")
                encoder = self._gpu_encoder(ext)
                if encoder:
                    encode_cmd.extend(["-c:v", encoder, "-preset", "p4"])

            encode_cmd.append(output_path)
            subprocess.run(encode_cmd, check=True)

    def _run_ai_step(self, step: dict, input_dir: str, output_dir: str) -> None:
        """Execute a single AI processing step on a directory of frames.

        This method dispatches to engine-specific implementations.
        Override or extend this to add new AI engines.
        """
        engine = step["engine"]
        step_type = step["type"]

        # Engine registry — maps engine names to handler functions
        registry = self._get_ai_registry()

        key = f"{step_type}:{engine}"
        if key not in registry:
            available = [k for k in registry if k.startswith(f"{step_type}:")]
            raise NotImplementedError(
                f"AI engine '{engine}' for '{step_type}' is not yet implemented. "
                f"Available: {[k.split(':')[1] for k in available] or 'none (install revid[ai])'}"
            )

        registry[key](step, input_dir, output_dir)

    def _get_ai_registry(self) -> dict:
        """Return the registry of available AI engine handlers.

        Each entry maps "type:engine" to a callable(step, input_dir, output_dir).
        """
        registry = {}
        try:
            from revid.engines import get_registry
            registry.update(get_registry())
        except ImportError:
            pass
        return registry

    def _render_stabilized(self, output_path: str, gpu: bool = False) -> None:
        """Run 2-pass vidstab stabilization."""
        params = self._stabilize_params
        with tempfile.TemporaryDirectory() as tmpdir:
            transforms_file = os.path.join(tmpdir, "transforms.trf")

            # Pass 1: detect motion — apply existing video filters first, then detect
            detect_filter = f"vidstabdetect=shakiness={params['shakiness']}:accuracy={params['accuracy']}:result={transforms_file}"
            pass1_filters = list(self._video_filters) + [detect_filter]

            cmd1 = ["ffmpeg", "-y"]
            cmd1.extend(self._pre_args)
            cmd1.extend(["-i", self._path])
            cmd1.extend(["-vf", ",".join(pass1_filters)])
            cmd1.extend(["-f", "null", "-"])
            subprocess.run(cmd1, check=True)

            # Pass 2: apply transforms + any remaining encoding
            transform_filter = f"vidstabtransform=input={transforms_file}:smoothing={params['smoothing']}"
            pass2_filters = list(self._video_filters) + [transform_filter]

            cmd2 = ["ffmpeg", "-y"]
            cmd2.extend(self._pre_args)
            cmd2.extend(["-i", self._path])
            cmd2.extend(["-vf", ",".join(pass2_filters)])

            if self._audio_filters:
                cmd2.extend(["-af", ",".join(self._audio_filters)])
            else:
                cmd2.extend(["-c:a", "copy"])

            if gpu:
                ext = os.path.splitext(output_path)[1].lstrip(".")
                encoder = self._gpu_encoder(ext)
                if encoder:
                    cmd2.extend(["-c:v", encoder, "-preset", "p4"])

            cmd2.extend(self._post_args)
            cmd2.append(output_path)
            subprocess.run(cmd2, check=True)

    def preview(self, at: float = 0.0, output: str | None = None) -> str:
        """Extract a single frame at the given timestamp with all video filters applied."""
        if output is None:
            name = os.path.splitext(os.path.basename(self._path))[0]
            output = os.path.join(os.path.dirname(self._path), f"{name}_preview.png")
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        cmd = ["ffmpeg", "-y", "-ss", str(at), "-i", self._path, "-frames:v", "1"]
        if self._video_filters:
            cmd.extend(["-vf", ",".join(self._video_filters)])
        cmd.append(output)

        subprocess.run(cmd, check=True)
        return output

    # -------------------------------------------------------------------------
    # Format export helpers
    # -------------------------------------------------------------------------

    def to(self, fmt: str, output: str | None = None, gpu: bool = False) -> str:
        """Generic export — dispatches to to_<fmt>() if it exists, otherwise renders directly."""
        method_name = f"to_{fmt}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(output, gpu=gpu)
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {fmt}. Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}")
        return self.render(output, fmt=fmt, gpu=gpu)

    def to_mp4(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="mp4", gpu=gpu)

    def to_avi(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="avi", gpu=gpu)

    def to_mov(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="mov", gpu=gpu)

    def to_mkv(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="mkv", gpu=gpu)

    def to_webm(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="webm", gpu=gpu)

    def to_flv(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="flv", gpu=gpu)

    def to_ts(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="ts", gpu=gpu)

    def to_mpeg(self, output: str | None = None, gpu: bool = False) -> str:
        return self.render(output, fmt="mpeg", gpu=gpu)

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    def info(self) -> dict:
        """Return a summary of the video metadata."""
        probe = self._probe()
        fmt = probe.get("format", {})
        vs = self._video_stream()
        aus = self._audio_stream()

        result = {
            "path": self._path,
            "format": fmt.get("format_long_name"),
            "duration": self.duration,
            "size_bytes": int(fmt["size"]) if "size" in fmt else None,
            "bitrate": int(fmt["bit_rate"]) if "bit_rate" in fmt else None,
        }

        if vs:
            result["video"] = {
                "codec": vs.get("codec_name"),
                "width": int(vs["width"]) if "width" in vs else None,
                "height": int(vs["height"]) if "height" in vs else None,
                "fps": self.fps,
                "pixel_format": vs.get("pix_fmt"),
            }

        if aus:
            result["audio"] = {
                "codec": aus.get("codec_name"),
                "sample_rate": int(aus["sample_rate"]) if "sample_rate" in aus else None,
                "channels": int(aus["channels"]) if "channels" in aus else None,
                "bitrate": int(aus["bit_rate"]) if "bit_rate" in aus else None,
            }

        return result

    # -------------------------------------------------------------------------
    # Extract audio / mute
    # -------------------------------------------------------------------------

    def extract_audio(self, output: str, fmt: str = "mp3") -> str:
        """Extract the audio track to a separate file."""
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        cmd = ["ffmpeg", "-y", "-i", self._path, "-vn"]
        if self._audio_filters:
            cmd.extend(["-af", ",".join(self._audio_filters)])
        cmd.append(output)

        subprocess.run(cmd, check=True)
        return output

    def mute(self) -> "BaseVideo":
        """Remove audio from the video."""
        clone = self._clone()
        clone._post_args.extend(["-an"])
        return clone

    # -------------------------------------------------------------------------
    # Concat
    # -------------------------------------------------------------------------

    @classmethod
    def concat(cls, *videos: "BaseVideo", output: str | None = None, fmt: str = "mp4") -> str:
        """Concatenate multiple videos into one.

        Args:
            videos: VideoFile instances to join (in order).
            output: Output file path.
            fmt: Output format (default "mp4").
        """
        if not videos:
            raise ValueError("At least one video is required")

        if output is None:
            name = os.path.splitext(os.path.basename(videos[0]._path))[0]
            directory = os.path.dirname(videos[0]._path)
            output = os.path.join(directory, f"{name}_concat.{fmt}")
        output = os.path.abspath(output)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for v in videos:
                f.write(f"file '{v._path}'\n")
            list_path = f.name

        try:
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output]
            subprocess.run(cmd, check=True)
        finally:
            os.unlink(list_path)

        return output

    # -------------------------------------------------------------------------
    # FFmpeg availability check
    # -------------------------------------------------------------------------

    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if ffmpeg and ffprobe are available."""
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                return False
        return True
