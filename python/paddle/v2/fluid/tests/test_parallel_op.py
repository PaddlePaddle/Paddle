import unittest

import paddle.v2.fluid.layers as layers
import paddle.v2.fluid as fluid
from paddle.v2.fluid.framework import Program
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.backward import append_backward
import numpy as np
import paddle.v2.fluid.core as core


class ParallelOpTest(unittest.TestCase):
    def setUp(self):
        x = layers.data(
            shape=[-1, 3, 4],
            dtype='float32',
            name='x',
            append_batch_size=False,
            stop_gradient=False)

        places = fluid.default_main_program().global_block().create_var()
        pd = layers.ParallelDo(places=places)

        with pd.do():
            data = pd.read_input(x)
            hidden = layers.fc(input=data, size=7)
            pd.write_output(hidden)
        data = pd()
        loss = layers.mean(x=data)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(loss)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        exe.run(fluid.default_main_program(),
                feed={
                    x.name: np.random.uniform(0.1, 0.6,
                                              (2, 3, 4)).astype("float32")
                })

    def test_forward(self):
        pass


if __name__ == '__main__':
    unittest.main()
