import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from gng import GrowingNeuralGas, GNGConfiguration


def generate_dataset(seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a richer 2D dataset mixing Gaussian clusters and a noisy ring so
    the learned topology has more structure to align with.
    """
    rng = np.random.default_rng(seed)
    points_per_cluster = 400
    clusters = []
    labels = []

    gaussian_specs = [
        ((-6.0, -2.0), 0.75),
        ((-2.5, 4.5), 0.85),
        ((3.0, 5.0), 0.55),
        ((6.0, -4.0), 0.65),
    ]

    for label, (center, scale) in enumerate(gaussian_specs):
        center_arr = np.asarray(center, dtype=np.float64)
        cluster = rng.normal(loc=center_arr, scale=scale, size=(points_per_cluster, 2))
        clusters.append(cluster)
        labels.append(np.full(cluster.shape[0], label))

    ring_points = 650
    theta = rng.uniform(0, 2 * np.pi, ring_points)
    radius = rng.normal(4.8, 0.25, ring_points)
    ring = np.column_stack(
        (
            np.cos(theta) * radius + 0.5,
            np.sin(theta) * radius - 0.5,
        )
    )
    clusters.append(ring)
    labels.append(np.full(ring.shape[0], len(gaussian_specs)))

    data = np.vstack(clusters)
    label_array = np.concatenate(labels)
    return data, label_array


def build_config(dim: int) -> GNGConfiguration:
    config = GNGConfiguration()
    config.dim = dim
    config.max_nodes = 20
    config.max_age = 75
    config.lambda_ = 200
    config.alpha = 0.45
    config.beta = 0.002
    config.eps_w = 0.06
    config.eps_n = 0.008
    return config


def plot_topology(data: np.ndarray, labels: np.ndarray, nodes) -> None:
    if not nodes:
        print("No nodes available to visualize.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.35,
        linewidths=0,
        label="Samples",
    )

    node_positions = np.array([node.position for node in nodes])
    ax.scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        s=120,
        c="crimson",
        edgecolors="white",
        linewidths=0.8,
        label="GNG nodes",
        zorder=3,
    )

    # Build undirected edges only once so the topology looks clean.
    point_map = {node.index: np.asarray(node.position) for node in nodes}
    edges = []
    for node in nodes:
        start = point_map[node.index]
        for neighbour_idx in node.neighbours:
            if neighbour_idx <= node.index:
                continue
            end = point_map.get(neighbour_idx)
            if end is None:
                continue
            edges.append([start, end])

    if edges:
        line_collection = LineCollection(
            edges,
            colors="royalblue",
            linewidths=1.2,
            alpha=0.7,
            zorder=2,
        )
        ax.add_collection(line_collection)

    ax.legend(loc="upper right")
    ax.set_title("Growing Neural Gas Topology")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linestyle="--")
    fig.colorbar(scatter, ax=ax, label="Source cluster")
    plt.tight_layout()
    plt.show()


def main():
    print("=== Growing Neural Gas Example ===\n")
    data, labels = generate_dataset()
    print(f"Generated {len(data):,} data points in 2D across {len(np.unique(labels))} regions")
    print(
        f"  X range: [{data[:, 0].min():.2f}, {data[:, 0].max():.2f}] | "
        f"Y range: [{data[:, 1].min():.2f}, {data[:, 1].max():.2f}]\n"
    )

    config = build_config(dim=data.shape[1])
    print("GNG Configuration:")
    print(f"  Dimensions: {config.dim}")
    print(f"  Max nodes: {config.max_nodes}")
    print(f"  Lambda (insertion interval): {config.lambda_}")
    print(f"  Max edge age: {config.max_age}")
    print(f"  Alpha: {config.alpha}, Beta: {config.beta}")
    print(f"  Eps_w: {config.eps_w}, Eps_n: {config.eps_n}\n")

    config.check()
    gng = GrowingNeuralGas(config=config, auto_run=False)

    try:
        gng.insert(data)
        print(f"Inserted {len(data)} training samples")

        print("\nTraining...")
        gng.run()
        time.sleep(3.0)
        gng.pause()

        nodes = gng.nodes()
        print(f"\n=== Results ===")
        print(f"Trained {len(nodes)} nodes")
        print(f"Mean error: {gng.mean_error():.4f}")

        print(f"\nFirst 5 node positions:")
        for node in nodes[:5]:
            pos = node.position
            print(
                f"  Node {node.index}: pos=[{pos[0]:6.2f}, {pos[1]:6.2f}], "
                f"error={node.error:.4f}, neighbors={node.neighbours}"
            )

        test_point = np.array([0.0, 0.0])
        predicted_node = gng.predict(test_point)
        test_point2 = np.array([5.0, 5.0])
        predicted_node2 = gng.predict(test_point2)

        print(f"\nPrediction test:")
        print(f"  Point {test_point} -> Node {predicted_node}")
        print(f"  Point {test_point2} -> Node {predicted_node2}")

        plot_topology(data, labels, nodes)
    finally:
        gng.terminate()
        print("\n=== Success! ===")


if __name__ == "__main__":
    main()
