import unittest
import paddle.v2.framework.create_op_creation_methods as creation
import paddle.v2.framework.core as core
import numpy


class TestFillZerosLikeOp(unittest.TestCase):
    def test_fill(self):
        scope = core.Scope()
        a = scope.create_var("input")
        a_tensor = a.get_tensor()
        a_tensor.set_dims([546, 291])
        a_tensor.alloc_float()
        a_tensor.set(numpy.random.random((546, 291)).astype("float32"))

        op = creation.op_creations.fill_zeros_like(Src="input", Dst="output")

        for out in op.outputs():
            if scope.get_var(out) is None:
                scope.create_var(out).get_tensor()

        b_tensor = scope.get_var("output").get_tensor()
        op.infer_shape(scope)
        self.assertEqual([546, 291], b_tensor.shape())
        ctx = core.DeviceContext.cpu_context()
        op.run(scope, ctx)
        b_tensor_array = numpy.array(b_tensor)
        for r in range(0, 546):
            for c in range(0, 291):
                self.assertEqual(b_tensor_array[r][c], 0.0)


if __name__ == '__main__':
    unittest.main()
