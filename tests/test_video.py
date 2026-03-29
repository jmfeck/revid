# -*- coding: utf-8 -*-

"""Test VideoFile class and filter chaining."""

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
        "testsrc=duration=2:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
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


class TestVideoFile:
    def test_read(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        assert video is not None
        assert video.path == os.path.abspath(sample_video)

    def test_read_format_specific(self, sample_video):
        import revid as vr

        video = vr.read_mp4(sample_video)
        assert video is not None

    def test_file_not_found(self):
        import revid as vr

        with pytest.raises(FileNotFoundError):
            vr.read("nonexistent_file.mp4")

    def test_repr(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        r = repr(video)
        assert "VideoFile" in r
        assert "mp4" in r
        assert "filters=0" in r

    def test_context_manager(self, sample_video):
        import revid as vr

        with vr.read(sample_video) as video:
            assert video is not None

    def test_properties(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        assert video.duration is not None
        assert video.duration > 0
        assert video.size == (320, 240)
        assert video.width == 320
        assert video.height == 240
        assert video.fps is not None

    def test_info(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        info = video.info()
        assert "video" in info
        assert "audio" in info
        assert info["video"]["width"] == 320
        assert info["video"]["height"] == 240
        assert info["video"]["codec"] == "h264"


class TestFilterChaining:
    def test_immutable_chaining(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        denoised = video.denoise(strength=0.5)
        assert len(video._video_filters) == 0
        assert len(denoised._video_filters) == 1

    def test_multiple_filters(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        result = video.deinterlace().denoise().sharpen().upscale(factor=2)
        assert len(result._video_filters) == 4

    def test_audio_filters(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        result = video.audio_denoise().hum_remove(frequency=60).hiss_remove()
        assert len(result._audio_filters) == 4  # hiss_remove adds highpass + lowpass

    def test_mixed_filters(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        result = video.denoise().audio_denoise()
        assert len(result._video_filters) == 1
        assert len(result._audio_filters) == 1

    def test_branching(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        deinterlaced = video.deinterlace()
        branch_a = deinterlaced.denoise(strength=0.3)
        branch_b = deinterlaced.denoise(strength=0.8)
        assert len(branch_a._video_filters) == 2
        assert len(branch_b._video_filters) == 2
        assert branch_a._video_filters[1] != branch_b._video_filters[1]


class TestRender:
    def test_render_basic(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out = f.name
        try:
            result = video.denoise(strength=0.3).to_mp4(out)
            assert os.path.isfile(result)
            assert os.path.getsize(result) > 0
        finally:
            os.unlink(out)

    def test_render_chain(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out = f.name
        try:
            video.deinterlace().denoise(strength=0.3).sharpen(amount=0.5).to_mp4(out)
            assert os.path.isfile(out)
            assert os.path.getsize(out) > 0
        finally:
            os.unlink(out)

    def test_preview(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out = f.name
        try:
            result = video.denoise().preview(at=0.5, output=out)
            assert os.path.isfile(result)
            assert os.path.getsize(result) > 0
        finally:
            os.unlink(out)

    def test_to_generic(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
            out = f.name
        try:
            video.to("mkv", out)
            assert os.path.isfile(out)
            assert os.path.getsize(out) > 0
        finally:
            os.unlink(out)

    def test_extract_audio(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out = f.name
        try:
            result = video.extract_audio(out)
            assert os.path.isfile(result)
            assert os.path.getsize(result) > 0
        finally:
            os.unlink(out)

    def test_mute(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        muted = video.mute()
        assert "-an" in muted._post_args

    def test_trim(self, sample_video):
        import revid as vr

        video = vr.read(sample_video)
        trimmed = video.trim(start=0.5, end=1.5)
        assert "-ss" in trimmed._pre_args
        assert "-to" in trimmed._pre_args


class TestFilters:
    """Test that each filter method produces valid filter strings."""

    def test_deinterlace_yadif(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).deinterlace()
        assert any("yadif" in f for f in v._video_filters)

    def test_deinterlace_bwdif(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).deinterlace(algorithm="bwdif")
        assert any("bwdif" in f for f in v._video_filters)

    def test_denoise_hqdn3d(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).denoise(strength=0.5)
        assert any("hqdn3d" in f for f in v._video_filters)

    def test_denoise_nlmeans(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).denoise(algorithm="nlmeans")
        assert any("nlmeans" in f for f in v._video_filters)

    def test_sharpen_unsharp(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).sharpen(amount=1.0)
        assert any("unsharp" in f for f in v._video_filters)

    def test_sharpen_cas(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).sharpen(algorithm="cas")
        assert any("cas" in f for f in v._video_filters)

    def test_upscale(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).upscale(factor=2)
        assert any("scale" in f for f in v._video_filters)

    def test_upscale_bicubic(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).upscale(factor=2, algorithm="bicubic")
        assert any("bicubic" in f for f in v._video_filters)

    def test_color_correct(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).color_correct(brightness=0.1, saturation=1.2)
        assert any("eq=" in f for f in v._video_filters)

    def test_white_balance(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).white_balance(temperature=5000)
        assert any("colortemperature" in f for f in v._video_filters)

    def test_deflicker(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).deflicker()
        assert any("deflicker" in f for f in v._video_filters)

    def test_crop(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).crop(x=10, y=10, width=300, height=220)
        assert any("crop" in f for f in v._video_filters)

    def test_set_fps(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).set_fps(24)
        assert any("fps=24" in f for f in v._video_filters)

    def test_rotate(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).rotate(90)
        assert any("transpose" in f for f in v._video_filters)

    def test_grayscale(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).grayscale()
        assert any("gray" in f for f in v._video_filters)

    def test_fade_in(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).fade_in(2.0)
        assert any("fade" in f for f in v._video_filters)

    def test_deblock(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).deblock()
        assert any("deblock" in f for f in v._video_filters)

    def test_hum_remove(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).hum_remove(frequency=60)
        assert any("bandreject" in f for f in v._audio_filters)

    def test_hum_remove_harmonics(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).hum_remove(frequency=60, harmonics=5)
        assert len(v._audio_filters) == 5

    def test_audio_normalize(self, sample_video):
        import revid as vr

        v = vr.read(sample_video).audio_normalize()
        assert any("loudnorm" in f for f in v._audio_filters)

    def test_invalid_denoise_algorithm(self, sample_video):
        import revid as vr

        with pytest.raises(ValueError):
            vr.read(sample_video).denoise(algorithm="invalid")

    def test_invalid_engine(self, sample_video):
        import revid as vr

        with pytest.raises(ValueError):
            vr.read(sample_video).upscale(engine="invalid_engine")
