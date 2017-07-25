import unittest
import paddle.v2.framework.create_op_creation_methods as creation
import paddle.v2.framework.core as core
from op_test_util import OpTestMeta
import numpy


class TestRandomOp(unittest.TestCase):
    def test_random(self):
        scope = core.Scope(None)
        # Out = scope.create_var("Out")
        op = creation.op_creations.random(
            shape=[1000, 1000], mean=5.0, std=1.0, seed=1701, Out="Out")
        for out in op.outputs():
            if scope.get_var(out) is None:
                scope.create_var(out).get_tensor()

        tensor = scope.get_var("Y").get_tensor()
        op.infer_shape(scope)
        self.assertEqual([1000, 1000], tensor.shape())
        ctx = core.DeviceContext.cpu_context()
        op.run(scope, ctx)
        self.assertAlmostEqual(numpy.std(tensor), 1.0)
        self.assertAlmostEqual(numpy.mean(tensor), 5.0)


if __name__ == '__main__':
    unittest.main()
