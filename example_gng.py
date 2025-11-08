import numpy as np
from gng import GrowingNeuralGas, GNGConfiguration

print("=== Growing Neural Gas Example ===\n")

# Generate sample 2D data (two clusters)
np.random.seed(42)
cluster1 = np.random.randn(100, 2) + [0, 0]
cluster2 = np.random.randn(100, 2) + [5, 5]
data = np.vstack([cluster1, cluster2])

print(f"Generated {len(data)} data points in 2D")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]\n")

# Create and configure GNG
config = GNGConfiguration()
config.dim = 2
config.max_nodes = 50
config.max_age = 88
config.lambda_ = 300
config.alpha = 0.5
config.beta = 0.0005
config.eps_w = 0.05
config.eps_n = 0.006

print("GNG Configuration:")
print(f"  Dimensions: {config.dim}")
print(f"  Max nodes: {config.max_nodes}")
print(f"  Lambda (insertion interval): {config.lambda_}")
print(f"  Max edge age: {config.max_age}")
print(f"  Alpha: {config.alpha}, Beta: {config.beta}")
print(f"  Eps_w: {config.eps_w}, Eps_n: {config.eps_n}\n")

# Check configuration validity
config.check()

# Create GNG instance (auto_run=False to control execution manually)
gng = GrowingNeuralGas(config=config, auto_run=False)

# Add data
gng.insert(data)
print(f"Inserted {len(data)} training samples")

# Run training
print("\nTraining...")
gng.run()

# Let it train for a bit
import time
time.sleep(2.0)

# Pause training to inspect results
gng.pause()

# Get results
nodes = gng.nodes()
print(f"\n=== Results ===")
print(f"Trained {len(nodes)} nodes")
print(f"Mean error: {gng.mean_error():.4f}")

print(f"\nFirst 5 node positions:")
for i, node in enumerate(nodes[:5]):
    pos = node.position
    print(f"  Node {node.index}: pos=[{pos[0]:6.2f}, {pos[1]:6.2f}], error={node.error:.4f}, neighbors={node.neighbours}")

# Test prediction
test_point = np.array([0.0, 0.0])
predicted_node = gng.predict(test_point)
print(f"\nPrediction test:")
print(f"  Point {test_point} -> Node {predicted_node}")

test_point2 = np.array([5.0, 5.0])
predicted_node2 = gng.predict(test_point2)
print(f"  Point {test_point2} -> Node {predicted_node2}")

# Clean up
gng.terminate()

print("\n=== Success! ===")
