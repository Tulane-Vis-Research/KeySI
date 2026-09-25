"""Assemble the KeySI Dash application."""

from .ui import app
from . import callbacks as _callbacks

__all__ = ["app"]
