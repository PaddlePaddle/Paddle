import unittest
import paddle.v2.framework.create_op_creation_methods as creation
import paddle.v2.framework.core as core
import numpy


class TestSgdOp(unittest.TestCase):
    def test_run(self):
        scope = core.Scope(None)
        # 1, set param
        param = scope.create_var("param")
        param_tensor = param.get_tensor()

        param_tensor.set_dims([2, 2])
        param_tensor.alloc_float()

        # param_data = numpy.random.random((2, 2)).astype("float32")
        param_data = numpy.ones((2, 2)).astype("float32")
        print param_data
        param_tensor.set(param_data)

        # 2. set grad
        param = scope.create_var("grad")
        grad_tensor = param.get_tensor()

        grad_tensor.set_dims([2])
        grad_tensor.alloc_float()

        # grad_data = numpy.random.random((2, 2)).astype("float32")
        grad_data = numpy.ones((2, 2)).astype("float32")
        print grad_data
        grad_tensor.set(grad_data)

        param_out = scope.create_var("param_out")

        sgd_op = creation.op_creations.sgd(param="param",
                                           grad="grad",
                                           param_out="param_out")
        print str(sgd_op)
        sgd_op.infer_shape(scope)
        ctx = core.DeviceContext.cpu_context()

        sgd_op.run(scope, ctx)

        param_out_tensor = numpy.array(scope.get_var("param_out").get_tensor())
        print(param_out_tensor)


if __name__ == "__main__":
    unittest.main()
