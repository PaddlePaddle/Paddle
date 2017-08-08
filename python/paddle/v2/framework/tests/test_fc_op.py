import unittest
import numpy as np
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


class TestFc(unittest.TestCase):
    def setUp(self):
        self.x_np_data = np.random.random((1000, 784))
        self.W_np_data = np.random.random((784, 100))

    def test_fc(self):
        scope = core.Scope()
        place = core.CPUPlace()
        x_tensor = scope.new_var("X").get_tensor()
        x_tensor.set_dims(self.x_np_data.shape)
        x_tensor.set(self.x_np_data, place)

        W_tensor = scope.new_var("W").get_tensor()
        W_tensor.set_dims(self.W_np_data.shape)
        W_tensor.set(self.W_np_data, place)

        op = Operator("fc", X="X", Y="Y", W="W")

        for out in op.outputs():
            if scope.find_var(out) is None:
                scope.new_var(out).get_tensor()

        Y_tensor = scope.find_var("Y").get_tensor()
        op.infer_shape(scope)
        self.assertEqual([1000, 100], Y_tensor.shape())

        ctx = core.DeviceContext.create(place)

        op.run(scope, ctx)

        py_data = np.matmul(self.x_np_data, self.W_np_data)
        op_data = np.array(Y_tensor)
        print py_data - op_data
        self.assertTrue(np.allclose(py_data, op_data))




if __name__ == '__main__':
    unittest.main()
