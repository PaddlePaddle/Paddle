import unittest
import paddle.v2.framework.create_op_creation_methods as creation
import paddle.v2.framework.core as core
from op_test_util import OpTestMeta
import numpy


class TestRandomOp(unittest.TestCase):
    def test_random(self):
        scope = core.Scope(None)
        # Out = scope.create_var("Out")
        op = creation.op_creations.gaussian_random(
            shape=[1000, 1000], mean=5.0, std=1.0, Out="Out")
        for out in op.outputs():
            if scope.get_var(out) is None:
                scope.create_var(out).get_tensor()

        tensor = scope.get_var("Out").get_tensor()
        op.infer_shape(scope)
        self.assertEqual([1000, 1000], tensor.shape())
        ctx = core.DeviceContext.cpu_context()
        op.run(scope, ctx)
        tensor_array = numpy.array(tensor)
        self.assertAlmostEqual(numpy.mean(tensor_array), 5.0, places=3)
        self.assertAlmostEqual(numpy.std(tensor_array), 1.0, places=3)


if __name__ == '__main__':
    unittest.main()
