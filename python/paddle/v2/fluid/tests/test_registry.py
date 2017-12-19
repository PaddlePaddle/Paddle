import unittest
import warnings

import paddle.v2.fluid as fluid
import paddle.v2.fluid.framework as framework
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.registry as registry


class TestRegistry(unittest.TestCase):
    def test_registry_layer(self):
        self.layer_type = "mean"
        program = framework.Program()

        x = fluid.layers.data(name='X', shape=[10, 10], dtype='float32')
        output = layers.mean(x)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        X = np.random.random((10, 10)).astype("float32")
        mean_out = exe.run(program, feed={"X": X}, fetch_list=[output])
        self.assertAlmostEqual(np.mean(X), mean_out)
