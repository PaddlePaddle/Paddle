import unittest
import numpy as np
from paddle.v2.fluid.executor import Executor
import paddle.v2.fluid.core as core
import paddle.v2.fluid.layers as pd


class TestSumOp(unittest.TestCase):
    def test_tensor(self):
        i = pd.zeros(shape=[2, 10], dtype='float32')

        pd.Print(i, message="I am a message", summarize=10)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        exe.run()


if __name__ == '__main__':
    unittest.main()
