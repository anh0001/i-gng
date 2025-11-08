"""Python package exposing the Growing Neural Gas bindings."""

from __future__ import annotations

try:
    from ._core import GNGConfiguration, GNGServer, load_gng
except ImportError as exc:  # pragma: no cover - surfaces during broken builds
    raise ImportError(
        "gng._core extension could not be imported. "
        "Build the package in editable/development mode (`pip install -e .`)."
    ) from exc

from .wrapper import GrowingNeuralGas

__all__ = [
    "GNGConfiguration",
    "GNGServer",
    "GrowingNeuralGas",
    "load_gng",
]

