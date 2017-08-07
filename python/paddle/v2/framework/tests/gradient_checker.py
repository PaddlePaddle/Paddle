import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator
import numpy
import unittest

__all__ = ['get_numeric_gradient']


def get_numeric_gradient(op,
                         input_values,
                         output_name,
                         input_to_check,
                         delta=1e-2,
                         local_scope=None):
    """
    Get Numeric Gradient for an operator's input.
    
    :param op: C++ operator instance, could be an network 
    :param input_values: The input variables. Should be an dictionary, key is 
    variable name. Value is numpy array.
    :param output_name: The final output variable name. 
    :param input_to_check: The input variable need to get gradient.
    :param delta: The perturbation value for numeric gradient method. The 
    smaller delta is, the more accurate result will get. But if that delta is
     too small, it could occur numerical stability problem.
    :param local_scope: The local scope used for get_numeric_gradient.
    :return: The gradient array in numpy format.
    """
    if local_scope is None:
        local_scope = core.Scope()

    # Create all input variable in local_scope
    for var_name in input_values:
        var = local_scope.new_var(var_name)
        tensor = var.get_tensor()
        tensor.set_dims(input_values[var_name].shape)
        tensor.alloc_float(core.CPUPlace())
        tensor.set(input_values[var_name], core.CPUPlace())

    # Create all output variable in local_scope
    for output in op.outputs():
        if local_scope.find_var(output) is None:
            local_scope.new_var(output).get_tensor()

    op.infer_shape(local_scope)

    # allocate output memory
    for output in op.outputs():
        local_scope.find_var(output).get_tensor().alloc_float(core.CPUPlace())

    # TODO(yuyang18): Only CPU is support now.
    cpu_ctx = core.DeviceContext.create(core.CPUPlace())

    def get_output():
        op.run(local_scope, cpu_ctx)
        return numpy.array(local_scope.find_var(output_name).get_tensor()).sum()

    def product(dim):
        return reduce(lambda a, b: a * b, dim, 1)

    tensor_to_check = local_scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.get_dims())
    gradient_flat = numpy.zeros(shape=(tensor_size, ), dtype='float32')
    for i in xrange(tensor_size):
        origin = tensor_to_check.get_float_element(i)
        x_pos = origin + delta
        tensor_to_check.set_float_element(i, x_pos)
        y_pos = get_output()

        x_neg = origin - delta
        tensor_to_check.set_float_element(i, x_neg)
        y_neg = get_output()

        tensor_to_check.set_float_element(i, origin)  # restore old value
        gradient_flat[i] = (y_pos - y_neg) / delta / 2
    return gradient_flat.reshape(tensor_to_check.get_dims())


if __name__ == '__main__':

    class GetNumericGradientTest(unittest.TestCase):
        def test_add_op(self):
            add_op = Operator('add_two', X="X", Y="Y", Out="Z")
            x = numpy.random.random((10, 1)).astype("float32")
            y = numpy.random.random((10, 1)).astype("float32")

            arr = get_numeric_gradient(add_op, {'X': x, "Y": y}, 'Z', 'X')
            self.assertAlmostEqual(arr.mean(), 1.0, delta=1e-2)

    unittest.main()
