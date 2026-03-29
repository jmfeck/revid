# -*- coding: utf-8 -*-

"""Central registry for AI engine handlers.

Each handler is a function: (step: dict, input_dir: str, output_dir: str) -> None
The handler reads frames from input_dir, processes them, and writes to output_dir.

Registry keys are "type:engine" strings, e.g. "upscale:realesrgan".
"""

_REGISTRY: dict = {}


def register(step_type: str, engine: str):
    """Decorator to register an AI engine handler."""

    def decorator(fn):
        _REGISTRY[f"{step_type}:{engine}"] = fn
        return fn

    return decorator


def get_registry() -> dict:
    """Return the current registry, importing available engines."""
    # Import engine modules — each one registers itself via @register
    _try_import("revid.engines.realesrgan")
    _try_import("revid.engines.gfpgan")
    _try_import("revid.engines.rife")
    _try_import("revid.engines.nafnet")
    _try_import("revid.engines.deoldify")
    _try_import("revid.engines.upscale_extra")
    _try_import("revid.engines.inpaint")
    _try_import("revid.engines.scratch")
    _try_import("revid.engines.stabilize_ai")
    _try_import("revid.engines.audio")
    _try_import("revid.engines.scene")
    return dict(_REGISTRY)


def _try_import(module: str) -> None:
    """Try to import a module, silently skip if dependencies are missing."""
    try:
        __import__(module)
    except ImportError:
        pass
