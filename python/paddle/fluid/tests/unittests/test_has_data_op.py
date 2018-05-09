from paddle.fluid.op import Operator
import paddle.fluid.core as core
import unittest
import numpy as np


class BeamSearchOpTester(unittest.TestCase):
    def setUp(self):
        self.scope = core.Scope()
        self.scope.var('X')
        self.scope.var('Out')
        self.place = core.CUDAPlace(0)
        x_data = np.array([])
        x_tensor = self.scope.var('X').get_tensor()
        x_tensor.set(x_data, self.place)
        out_tensor = self.scope.var('Out').get_tensor()

    def test_run(self):
        op = Operator('has_data', X='X', Out='Out')
        op.run(self.scope, self.place)
        out_tensor = self.scope.find_var('Out').get_tensor()
        print 'output: ', np.array(out_tensor)


if __name__ == '__main__':
    unittest.main()
