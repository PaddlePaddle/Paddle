import paddle.v2.framework.core as core
import unittest
import numpy
import paddle.v2.framework.create_op_creation_methods as creation
from gradient_checker import get_numeric_gradient


def create_op(op_type):
    func = getattr(creation.op_creations, op_type, None)
    assert (func is not None)

    kwargs = dict()
    for in_name in func.all_input_args:
        kwargs[in_name] = in_name
    for out_name in func.all_output_args:
        kwargs[out_name] = out_name
    op = func(**kwargs)
    return op


def grad_var_name(var_name):
    return var_name + "@GRAD"


def label_softmax_grad(Y, dY):
    dX = Y * 0.0
    for i in range(Y.shape[0]):
        d = numpy.dot(Y[i, :], dY[i, :])
        dX[i, :] = Y[i, :] * (dY[i, :] - d)
    return dX


class GradChecker(unittest.TestCase):
    def assert_grad(self,
                    forward_op,
                    inputs=dict(),
                    input_to_check=set(),
                    no_grad_set=set(),
                    only_cpu=False):
        backward_op = core.Operator.backward(forward_op, no_grad_set)
        print(backward_op)

        #
        out_names = forward_op.outputs()
        if len(out_names) != 1:
            raise ValueError("out_names should be 1")

        in_names = backward_op.inputs()
        for no_grad in no_grad_set:
            if no_grad not in in_names:
                raise ValueError("no_grad should be in in_names")

        cpu_scope = core.Scope()
        cpu_place = core.CPUPlace()
        ctx = core.DeviceContext.create(cpu_place)

        # create input var and set value
        for name, value in inputs.iteritems():
            assert name in in_names
            var = cpu_scope.new_var(name).get_tensor()
            var.set_dims(value.shape)
            var.set(value, cpu_place)

        # create output var
        for out_name in forward_op.outputs():
            cpu_scope.new_var(out_name).get_tensor()

        # infer the shape of output var and set value of output var
        forward_op.infer_shape(cpu_scope)
        forward_op.run(cpu_scope, ctx)

        # create input grad var and set shape as the input var
        for name in forward_op.inputs():
            in_tensor = cpu_scope.find_var(name).get_tensor()
            grad_tensor = cpu_scope.new_var(grad_var_name(name)).get_tensor()
            grad_tensor.set_dims(in_tensor.shape())

        # create output grad var
        # set shape as the output var
        # set value of this grad to ones
        for name in forward_op.outputs():
            out_tensor = cpu_scope.find_var(name).get_tensor()
            grad_tensor = cpu_scope.new_var(grad_var_name(name)).get_tensor()
            grad_tensor.set_dims(out_tensor.shape())
            data = 1.0 * numpy.ones(out_tensor.shape())
            grad_tensor.set(data, cpu_place)

        backward_op.infer_shape(cpu_scope)
        backward_op.run(cpu_scope, ctx)

        numeric_input = dict()
        for name in backward_op.inputs():
            data = numpy.array(cpu_scope.find_var(name).get_tensor())
            numeric_input[name] = data
        output_name = forward_op.outputs()[0]
        input_to_check = forward_op.inputs()[0]

        ret_val = get_numeric_gradient(forward_op, numeric_input, output_name,
                                       input_to_check)

        out_data = numpy.array(
            cpu_scope.find_var(backward_op.outputs()[0]).get_tensor())

        numpy.testing.assert_almost_equal(out_data, ret_val)


class SoftmaxGradOpTest(GradChecker):
    def test_softmax(self):
        op = create_op("softmax")
        X = numpy.random.random((3, 4)).astype("float32")
        inputs = {"X": X}
        self.assert_grad(op, inputs)


if __name__ == '__main__':
    unittest.main()
