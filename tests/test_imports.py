# -*- coding: utf-8 -*-

"""Test that the package imports correctly."""


def test_version():
    import revid

    assert revid.__version__ == "0.1.2"


def test_read_function_exists():
    import revid

    assert callable(revid.read)


def test_videofile_class_exists():
    import revid

    assert revid.VideoFile is not None


def test_all_format_readers_exist():
    import revid

    formats = ["mp4", "avi", "mov", "mkv", "webm", "flv", "ogv", "wmv", "3gp", "ts", "mpeg", "mpg"]
    for fmt in formats:
        fn = getattr(revid, f"read_{fmt}")
        assert callable(fn), f"read_{fmt} is not callable"


def test_all_exports():
    import revid

    for name in revid.__all__:
        assert hasattr(revid, name), f"revid.__all__ lists '{name}' but it doesn't exist"
