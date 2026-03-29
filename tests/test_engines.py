# -*- coding: utf-8 -*-

"""Test engine registry and AI step integration."""

import os
import subprocess
import tempfile

import pytest


@pytest.fixture
def sample_video():
    """Create a minimal test video with FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield path
    os.unlink(path)


class TestRegistry:
    def test_registry_loads(self):
        from revid.engines.registry import get_registry
        registry = get_registry()
        assert isinstance(registry, dict)
        assert len(registry) > 0

    def test_expected_engines_registered(self):
        from revid.engines.registry import get_registry
        registry = get_registry()

        expected = [
            "upscale:realesrgan",
            "face_restore:gfpgan",
            "face_restore:codeformer",
            "interpolate:rife",
            "denoise:nafnet",
            "colorize:deoldify",
            "colorize:ddcolor",
            "deblur:nafnet",
            "inpaint:lama",
            "object_remove:propainter",
            "scratch_remove:rtn",
            "stabilize:raft",
            "audio_denoise:demucs",
            "audio_separate:demucs",
            "audio_upscale:audiosr",
            "scene_detect:pyscenedetect",
        ]

        for key in expected:
            assert key in registry, f"Engine '{key}' not registered"

    def test_engine_count(self):
        from revid.engines.registry import get_registry
        registry = get_registry()
        assert len(registry) >= 38

    def test_all_handlers_are_callable(self):
        from revid.engines.registry import get_registry
        registry = get_registry()
        for key, handler in registry.items():
            assert callable(handler), f"Handler for '{key}' is not callable"


class TestAIStepIntegration:
    def test_ai_upscale_adds_step(self, sample_video):
        import revid as vr
        video = vr.read(sample_video)
        result = video.upscale(factor=4, engine="realesrgan")
        assert len(result._ai_steps) == 1
        assert result._ai_steps[0]["type"] == "upscale"
        assert result._ai_steps[0]["engine"] == "realesrgan"
        assert result._ai_steps[0]["factor"] == 4

    def test_ai_denoise_adds_step(self, sample_video):
        import revid as vr
        video = vr.read(sample_video)
        result = video.denoise(engine="nafnet")
        assert len(result._ai_steps) == 1
        assert result._ai_steps[0]["engine"] == "nafnet"

    def test_ai_face_restore_adds_step(self, sample_video):
        import revid as vr
        video = vr.read(sample_video)
        result = video.face_restore(engine="gfpgan")
        assert len(result._ai_steps) == 1
        assert result._ai_steps[0]["type"] == "face_restore"

    def test_ai_colorize_adds_step(self, sample_video):
        import revid as vr
        video = vr.read(sample_video)
        result = video.colorize(engine="deoldify")
        assert len(result._ai_steps) == 1
        assert result._ai_steps[0]["type"] == "colorize"

    def test_mixed_ffmpeg_and_ai(self, sample_video):
        import revid as vr
        video = vr.read(sample_video)
        result = (video
            .deinterlace()
            .denoise(strength=0.5)
            .upscale(factor=4, engine="realesrgan")
            .face_restore(engine="gfpgan")
        )
        assert len(result._video_filters) == 2
        assert len(result._ai_steps) == 2

    def test_ai_audio_adds_step(self, sample_video):
        import revid as vr
        video = vr.read(sample_video)
        result = video.audio_denoise(engine="demucs")
        assert len(result._ai_steps) == 1
        assert result._ai_steps[0]["type"] == "audio_denoise"

    def test_post_ai_resize(self, sample_video):
        import revid as vr
        video = vr.read(sample_video)
        result = video.upscale(factor=2, engine="realesrgan").resize(320, 240)
        assert len(result._post_ai_filters) == 1
        assert "scale=320:240" in result._post_ai_filters[0]
