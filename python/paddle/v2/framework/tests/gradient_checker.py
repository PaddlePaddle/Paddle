import paddle.v2.framework.core as core
from paddle.v2.framework.create_op_creation_methods import op_creations
import numpy
import unittest


def get_numeric_gradient(op,
                         input_values,
                         output_name,
                         input_to_check,
                         delta=1e-5,
                         local_scope=None):
    if local_scope is None:
        local_scope = core.Scope()
    for var_name in input_values:
        var = local_scope.new_var(var_name)
        tensor = var.get_tensor()
        tensor.set_dims(input_values[var_name].shape)
        tensor.alloc_float()
        tensor.set(input_values[var_name])

    for output in op.outputs():
        local_scope.new_var(output).get_tensor()

    op.infer_shape(local_scope)

    for output in op.outputs():
        local_scope.find_var(output).get_tensor().alloc_float()

    cpu_ctx = core.DeviceContext.cpu_context()

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
            add_op = op_creations.add_two(X="X", Y="Y", Out="Z")
            x = numpy.random.random((10, 1)).astype("float32")
            y = numpy.random.random((10, 1)).astype("float32")

            arr = get_numeric_gradient(add_op, {'X': x, "Y": y}, 'Z', 'X')

            self.assertAlmostEqual(arr.mean(), 1.0, delta=1e-2)

    unittest.main()
