import unittest

import paddle.v2.fluid.layers as layers
import paddle.v2.fluid as fluid
from paddle.v2.fluid.framework import Program
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.backward import append_backward_ops
import numpy as np
import paddle.v2.fluid.core as core


class ParallelOpTest(unittest.TestCase):
    def setUp(self):
        x = layers.data(
            shape=[2, 3, 4], dtype='float32', name='x', append_batch_size=False)

        places = fluid.default_main_program().global_block().create_var()
        pd = layers.ParallelDo(places=places)

        with pd.do():
            data = pd.read_input(x)
            hidden = layers.fc(input=data, size=7)
            pd.write_output(hidden)
        data = pd()
        print data
        print fluid.default_main_program()

    def test_forward(self):
        pass


if __name__ == '__main__':
    unittest.main()
