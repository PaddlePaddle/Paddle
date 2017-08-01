import paddle.v2.framework.core as core
import unittest
import numpy
import paddle.v2.framework.create_op_creation_methods as creation


class TestFc(unittest.TestCase):
    def test_fc(self):
        scope = core.Scope()
        x = scope.new_var("X")
        x_tensor = x.get_tensor()
        x_tensor.set_dims([1000, 784])
        x_tensor.alloc_float()

        w = scope.new_var("W")
        w_tensor = w.get_tensor()
        w_tensor.set_dims([784, 100])
        w_tensor.alloc_float()

        w_tensor.set(numpy.random.random((784, 100)).astype("float32"))

        # Set a real numpy array here.
        # x_tensor.set(numpy.array([]))

        op = creation.op_creations.fc(X="X", Y="Y", W="W")

        for out in op.outputs():
            if scope.find_var(out) is None:
                scope.new_var(out).get_tensor()

        tensor = scope.find_var("Y").get_tensor()
        op.infer_shape(scope)
        self.assertEqual([1000, 100], tensor.shape())

        ctx = core.DeviceContext.cpu_context()

        op.run(scope, ctx)

        # After complete all ops, check Y is expect or not.


if __name__ == '__main__':
    unittest.main()
