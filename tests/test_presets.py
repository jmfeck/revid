# -*- coding: utf-8 -*-

"""Test presets."""

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
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    yield path
    os.unlink(path)


class TestPresets:
    def test_all_presets_exist(self):
        from revid.presets import PRESETS

        expected = [
            "vhs_standard",
            "vhs_quality",
            "dvd_cleanup",
            "camcorder",
            "film_8mm",
            "vhs_ai",
            "vhs_ai_full",
            "camcorder_ai",
            "film_8mm_ai",
            "bw_restore",
        ]
        for name in expected:
            assert name in PRESETS, f"Preset '{name}' not found"

    def test_preset_count(self):
        from revid.presets import PRESETS

        assert len(PRESETS) == 10

    def test_ffmpeg_presets_produce_filters(self, sample_video):
        import revid as rv

        video = rv.read(sample_video)
        ffmpeg_presets = ["vhs_standard", "vhs_quality", "dvd_cleanup", "camcorder", "film_8mm"]

        for name in ffmpeg_presets:
            result = video.preset(name)
            assert len(result._video_filters) > 0, f"Preset '{name}' produced no video filters"

    def test_ai_presets_produce_ai_steps(self, sample_video):
        import revid as rv

        video = rv.read(sample_video)
        ai_presets = ["vhs_ai", "vhs_ai_full", "camcorder_ai", "film_8mm_ai", "bw_restore"]

        for name in ai_presets:
            result = video.preset(name)
            assert len(result._ai_steps) > 0, f"Preset '{name}' produced no AI steps"

    def test_ffmpeg_preset_renders(self, sample_video):
        import revid as rv

        video = rv.read(sample_video)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out = f.name
        try:
            video.preset("dvd_cleanup").to_mp4(out)
            assert os.path.isfile(out)
            assert os.path.getsize(out) > 0
        finally:
            os.unlink(out)

    def test_invalid_preset(self, sample_video):
        import revid as rv

        video = rv.read(sample_video)
        with pytest.raises(ValueError):
            video.preset("nonexistent")

    def test_preset_immutability(self, sample_video):
        import revid as rv

        video = rv.read(sample_video)
        _ = video.preset("vhs_standard")
        assert len(video._video_filters) == 0
