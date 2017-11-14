import unittest
import numpy as np
from paddle.v2.framework.op import Operator
import paddle.v2.framework.core as core


def create_tensor(scope, name, np_data):
    tensor = scope.var(name).get_tensor()
    tensor.set_dims(np_data.shape)
    tensor.set(np_data, core.CPUPlace())
    return tensor


class TestIsEmptyOp(unittest.TestCase):
    def setUp(self):
        self.scope = core.Scope()
        # create input variables
        np_data0 = np.array([0, 1, 2])
        create_tensor(self.scope, "X0", np_data0)

        np_data1 = np.array([1])
        t = create_tensor(self.scope, "X1", np_data1)
        t.set_dims([0])

        # create output variables
        self.scope.var("out")

    def test_no_empty(self):
        self.one_case("X0", False)

    def test_empty(self):
        self.one_case("X1", True)

    def one_case(self, input, target):
        op = Operator(type="is_empty", X=input, Y="out")
        ctx = core.DeviceContext.create(core.CPUPlace())
        op.run(self.scope, ctx)
        out = self.scope.var("out").get_tensor()
        self.assertEqual(np.array(out)[0], target)


if __name__ == "__main__":
    unittest.main()
