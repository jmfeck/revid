# -*- coding: utf-8 -*-

"""AI engine registry for revid.

Each engine module registers its handlers here.
Handlers are functions that process a directory of frames.
"""

from revid.engines.registry import get_registry

__all__ = ["get_registry"]
