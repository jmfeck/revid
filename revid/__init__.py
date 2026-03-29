# -*- coding: utf-8 -*-

"""revid — Simple video restoration for Python."""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Lazy-loading API
# ---------------------------------------------------------------------------

_CLASS_MAP = {
    "VideoFile": ("revid.video", "VideoFile"),
}

_READER_MAP = {
    "read_mp4": ("revid.video", "VideoFile", ".mp4"),
    "read_avi": ("revid.video", "VideoFile", ".avi"),
    "read_mov": ("revid.video", "VideoFile", ".mov"),
    "read_mkv": ("revid.video", "VideoFile", ".mkv"),
    "read_webm": ("revid.video", "VideoFile", ".webm"),
    "read_flv": ("revid.video", "VideoFile", ".flv"),
    "read_ogv": ("revid.video", "VideoFile", ".ogv"),
    "read_wmv": ("revid.video", "VideoFile", ".wmv"),
    "read_3gp": ("revid.video", "VideoFile", ".3gp"),
    "read_ts": ("revid.video", "VideoFile", ".ts"),
    "read_mpeg": ("revid.video", "VideoFile", ".mpeg"),
    "read_mpg": ("revid.video", "VideoFile", ".mpg"),
}


def _make_reader(module_path: str, class_name: str):
    """Create a reader function that lazily imports the class."""

    def reader(path: str, **kwargs):
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(path, **kwargs)

    return reader


def read(path: str, **kwargs):
    """Read any supported video file. Detects format automatically."""
    import importlib

    mod = importlib.import_module("revid.video")
    return mod.VideoFile(path, **kwargs)


def __getattr__(name: str):
    # Classes
    if name in _CLASS_MAP:
        import importlib

        module_path, class_name = _CLASS_MAP[name]
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        globals()[name] = cls
        return cls

    # Reader functions
    if name in _READER_MAP:
        module_path, class_name, _ = _READER_MAP[name]
        fn = _make_reader(module_path, class_name)
        fn.__name__ = name
        fn.__doc__ = f"Read a {_READER_MAP[name][2]} file and return a VideoFile."
        globals()[name] = fn
        return fn

    raise AttributeError(f"module 'revid' has no attribute {name!r}")


__all__ = [
    "__version__",
    "read",
    "VideoFile",
    *_READER_MAP.keys(),
]
