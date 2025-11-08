import time

import numpy as np

from gng import GNGConfiguration, GrowingNeuralGas


def test_basic_training_cycle():
    config = GNGConfiguration()
    config.dim = 2
    config.max_nodes = 20
    config.uniformgrid_optimization = False

    model = GrowingNeuralGas(config, auto_run=False)
    points = np.random.random((200, config.dim))

    model.insert(points)
    model.run()
    time.sleep(0.05)
    model.pause()

    nodes = model.nodes()
    assert nodes, "Growing Neural Gas should create at least one node"

    prediction = model.predict(points[0])
    assert isinstance(prediction, int)

    model.terminate()

