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


class GradChecker(unittest.TestCase):
    def assert_grad(self,
                    forward_op,
                    inputs=dict(),
                    input_to_check=set(),
                    no_grad_set=set(),
                    only_cpu=False):
        out_names = filter(lambda name: name != "@TEMP@", forward_op.outputs())
        if len(out_names) != 1:
            raise ValueError("non empty out_names should be 1")

        in_names = forward_op.inputs()
        for no_grad in no_grad_set:
            if no_grad not in in_names:
                raise ValueError("no_grad should be in in_names")

        backward_op = core.Operator.backward(forward_op, no_grad_set)

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

        # create output grad var
        # set shape as the output var
        # set value of this grad to ones
        for name in forward_op.outputs():
            out_tensor = cpu_scope.find_var(name).get_tensor()
            grad_tensor = cpu_scope.new_var(grad_var_name(name)).get_tensor()
            grad_tensor.set_dims(out_tensor.shape())
            data = 1.0 * numpy.ones(out_tensor.shape())
            grad_tensor.set(data, cpu_place)

        # create input grad var
        for name in backward_op.outputs():
            cpu_scope.new_var(name).get_tensor()

        backward_op.infer_shape(cpu_scope)
        backward_op.run(cpu_scope, ctx)

        numeric_input = dict()
        for name in forward_op.inputs():
            data = numpy.array(cpu_scope.find_var(name).get_tensor())
            numeric_input[name] = data
        output_name = forward_op.outputs()[0]
        input_to_check = forward_op.inputs()[0]

        numeric_grad = get_numeric_gradient(forward_op, numeric_input,
                                            output_name, input_to_check)
        op_grad = numpy.array(
            cpu_scope.find_var(backward_op.outputs()[0]).get_tensor())

        numpy.testing.assert_almost_equal(numeric_grad, op_grad, decimal=1e-2)
