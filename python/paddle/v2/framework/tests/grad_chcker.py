import paddle.v2.framework.core as core
import unittest
import numpy
import paddle.v2.framework.create_op_creation_methods as creation


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


def gradvar_name(var_name):
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
        for name, value in forward_op.inputs().iteritems():
            assert name in in_names
            var = cpu_scope.new_var(name).get_tensor()
            var.set_dims(value.shape)
            var.set(value, cpu_place)

        # create output var
        for out_name in forward_op.outputs():
            cpu_scope.new_var(out_name).get_tensor()

        forward_op.infer_shape(cpu_scope)
        forward_op.run(cpu_scope, ctx)

        # create forward output grad var
        for name in forward_op.outputs():
            out_tensor = cpu_scope.find_var(name).get_tensor()
            grad_tensor = cpu_scope.new_var(gradvar_name(name)).get_tensor()
            grad_tensor.set_dims(out_tensor.shape())
            data = 1.0 * numpy.ones(out_tensor.shape())
            grad_tensor.set(data, cpu_place)

        backward_op.infer_shape(cpu_scope)
        backward_op.run(cpu_scope, ctx)

        out_data = numpy.array(
            cpu_scope.find_var(gradvar_name("X")).get_tensor())

        Y_data = numpy.array(cpu_scope.find_var("Y").get_tensor())
        dY_data = numpy.array(
            cpu_scope.find_var(gradvar_name("Y")).get_tensor())
        expect = label_softmax_grad(Y_data, dY_data)

        numpy.testing.assert_almost_equal(out_data, expect)


class SoftmaxGradOpTest(GradChecker):
    def test_add_two(self):
        op = create_op("softmax")
        X = numpy.random.random((100, 100)).astype("float32")
        inputs = {"X": X}
        self.assert_grad(op, inputs)


if __name__ == '__main__':
    unittest.main()
