from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ._core import GNGConfiguration, GNGServer


@dataclass
class NodeSnapshot:
    """Lightweight Python representation of a single GNG node."""

    index: int
    position: np.ndarray
    error: float
    label: float
    utility: float
    neighbours: Sequence[int]


class GrowingNeuralGas:
    """
    Small Python helper around the C++ `GNGServer`.

    It keeps the Python API ergonomic while delegating all heavy lifting to the
    compiled extension.
    """

    def __init__(
        self,
        config: Optional[GNGConfiguration] = None,
        auto_run: bool = True,
    ) -> None:
        self.config = config or GNGConfiguration()
        self.server = GNGServer(self.config)
        if auto_run:
            self.run()

    def insert(
        self,
        data: np.ndarray,
        labels: Optional[Iterable[float]] = None,
        probabilities: Optional[Iterable[float]] = None,
        run_after: bool = False,
    ) -> "GrowingNeuralGas":
        arr = np.ascontiguousarray(data, dtype=np.float64)
        lbl = (
            None if labels is None else np.ascontiguousarray(labels, dtype=np.float64)
        )
        prob = (
            None
            if probabilities is None
            else np.ascontiguousarray(probabilities, dtype=np.float64)
        )
        self.server.insert_examples(arr, lbl, prob)
        if run_after:
            self.run()
        return self

    def run(self) -> "GrowingNeuralGas":
        self.server.run()
        return self

    def pause(self) -> "GrowingNeuralGas":
        self.server.pause()
        return self

    def terminate(self) -> None:
        self.server.terminate()

    def predict(self, vector: Sequence[float]) -> int:
        arr = np.ascontiguousarray(vector, dtype=np.float64)
        return self.server.predict(arr)

    def mean_error(self) -> float:
        return self.server.mean_error()

    def nodes(self) -> Sequence[NodeSnapshot]:
        snapshots = []
        for node in self.server.nodes():
            snapshots.append(
                NodeSnapshot(
                    index=node["index"],
                    position=np.asarray(node["position"]),
                    error=node["error"],
                    label=node["label"],
                    utility=node["utility"],
                    neighbours=tuple(node["neighbours"]),
                )
            )
        return snapshots
