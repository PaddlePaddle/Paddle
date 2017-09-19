import unittest
import numpy as np
import itertools
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


def grad_var_name(var_name):
    return var_name + "@GRAD"


def create_op(scope, op_type, inputs, outputs, attrs):
    kwargs = dict()

    for in_name, in_dup in Operator.get_op_inputs(op_type):
        if in_name in inputs:
            kwargs[in_name] = []
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, _ in sub_in:
                    var = scope.new_var(sub_in_name)
                    kwargs[in_name].append(sub_in_name)
            else:
                var = scope.new_var(in_name)
                kwargs[in_name].append(in_name)

    for out_name, out_dup in Operator.get_op_outputs(op_type):
        if out_name in outputs:
            kwargs[out_name] = []
            if out_dup:
                sub_out = outputs[out_name]
                for sub_out_name, _ in sub_out:
                    var = scope.new_var(sub_out_name)
                    kwargs[out_name].append(sub_out_name)
            else:
                var = scope.new_var(out_name)
                kwargs[out_name].append(out_name)

    for attr_name in Operator.get_op_attr_names(op_type):
        if attr_name in attrs:
            kwargs[attr_name] = attrs[attr_name]

    return Operator(op_type, **kwargs)


def set_input(scope, op, inputs, place):
    for in_name, in_dup in Operator.get_op_inputs(op.type()):
        if in_name in inputs:
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, sub_in_val in sub_in:
                    var = scope.find_var(sub_in_name)
                    tensor = var.get_tensor()
                    sub_in_array = sub_in_val[0] \
                        if isinstance(sub_in_val, tuple) else sub_in_val
                    tensor.set_dims(sub_in_array.shape)
                    tensor.set(sub_in_array, place)
                    if isinstance(sub_in_val, tuple):
                        tensor.set_lod(sub_in_val[1])
            else:
                var = scope.find_var(in_name)
                tensor = var.get_tensor()
                in_val = inputs[in_name]
                in_array = in_val[0] if isinstance(in_val, tuple) else in_val
                tensor.set_dims(in_array.shape)
                tensor.set(in_array, place)
                if isinstance(in_val, tuple):
                    tensor.set_lod(in_val[1])


def set_output_grad(scope, op, outputs, place):
    for out_name, out_dup in Operator.get_op_outputs(op.type()):
        if out_name in outputs:
            if out_dup:
                sub_out = outputs[out_name]
                for sub_out_name, _ in sub_out:
                    out_tensor = scope.find_var(sub_out_name).get_tensor()
                    grad_tensor = scope.new_var(grad_var_name(
                        sub_out_name)).get_tensor()
                    grad_tensor.set_dims(out_tensor.shape())
                    data = np.ones(out_tensor.shape(), dtype=np.float32)
                    grad_tensor.set(data, place)
            else:
                out_tensor = scope.find_var(out_name).get_tensor()
                grad_tensor = scope.new_var(grad_var_name(out_name)).get_tensor(
                )
                grad_tensor.set_dims(out_tensor.shape())
                data = np.ones(out_tensor.shape(), dtype=np.float32)
                grad_tensor.set(data, place)


def get_numeric_gradient(scope,
                         op,
                         inputs,
                         input_to_check,
                         output_names,
                         delta=0.005,
                         in_place=False):

    set_input(scope, op, inputs, core.CPUPlace())
    op.infer_shape(scope)

    tensor_to_check = scope.find_var(input_to_check).get_tensor()

    def product(dim):
        return reduce(lambda a, b: a * b, dim, 1)

    ctx = core.DeviceContext.create(core.CPUPlace())

    def get_output():
        sum = 0.0
        for output_name in output_names:
            op.run(scope, ctx)
            sum += np.array(scope.find_var(output_name).get_tensor()).sum()
        return sum

    tensor_to_check = scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.get_dims())
    gradient_flat = np.zeros(shape=(tensor_size, ), dtype='float32')
    # we only compute gradient of one element each time.
    # we use a for loop to compute the gradient of every element.
    for i in xrange(tensor_size):
        if in_place:
            set_input(scope, op, inputs, core.CPUPlace())

        # get one input element throw it's index i.
        origin = tensor_to_check.get_float_element(i)
        # add delta to it, run op and then get the sum of the result tensor.
        x_pos = origin + delta
        tensor_to_check.set_float_element(i, x_pos)
        y_pos = get_output()

        if in_place:
            set_input(scope, op, inputs, core.CPUPlace())

        x_neg = origin - delta
        tensor_to_check.set_float_element(i, x_neg)
        y_neg = get_output()

        tensor_to_check.set_float_element(i, origin)
        gradient_flat[i] = (y_pos - y_neg) / delta / 2

    return gradient_flat.reshape(tensor_to_check.get_dims())


def get_backward_op(scope, op, no_grad_set):
    backward_op = core.Operator.backward(op, no_grad_set)
    for input in backward_op.input_vars():
        var = scope.new_var(input)
        var.get_tensor()
    for output in backward_op.output_vars():
        var = scope.new_var(output)
        var.get_tensor()
    return backward_op


def get_gradient(scope, op, inputs, outputs, grad_name, place,
                 no_grad_set=None):
    ctx = core.DeviceContext.create(place)

    set_input(scope, op, inputs, place)

    op.infer_shape(scope)
    op.run(scope, ctx)

    if no_grad_set is None:
        no_grad_set = set()

    backward_op = get_backward_op(scope, op, no_grad_set)
    set_output_grad(scope, op, outputs, place)

    backward_op.infer_shape(scope)
    backward_op.run(scope, ctx)

    out = np.array(scope.find_var(grad_name).get_tensor())
    return out


class OpTest(unittest.TestCase):
    def check_output_with_place(self, place):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()
        self.op = create_op(self.scope, self.op_type, op_inputs, op_outputs,
                            op_attrs)
        if isinstance(place, core.GPUPlace) and not self.op.support_gpu():
            return
        set_input(self.scope, self.op, self.inputs, place)
        self.op.infer_shape(self.scope)
        ctx = core.DeviceContext.create(place)
        self.op.run(self.scope, ctx)

        for out_name, out_dup in Operator.get_op_outputs(self.op.type()):
            if out_name not in self.outputs:
                continue

            if out_dup:
                sub_out = self.outputs[out_name]
                if not isinstance(sub_out, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))

                for sub_out_name, expect in sub_out:
                    actual = np.array(
                        self.scope.find_var(sub_out_name).get_tensor())
                    self.assertTrue(
                        np.allclose(
                            actual, expect, atol=1e-05),
                        "output name: " + out_name + " has diff")
            else:
                actual = np.array(self.scope.find_var(out_name).get_tensor())
                expect = self.outputs[out_name]
                self.assertTrue(
                    np.allclose(
                        actual, expect, atol=1e-05),
                    "output name: " + out_name + " has diff")

    def check_output(self):
        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))
        for place in places:
            self.check_output_with_place(place)

    def __assert_is_close(self, numeric_grads, analytic_grads, names,
                          max_relative_error, msg_prefix):

        for a, b, name in itertools.izip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return "%s Variable %s max gradient diff %f over limit %f, the first " \
                  "error element is %d" % (
                   msg_prefix, name, max_diff, max_relative_error, offset)

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   in_place=False,
                   max_relative_error=0.005):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()
        self.op = create_op(self.scope, self.op_type, op_inputs, op_outputs,
                            op_attrs)
        if no_grad_set is None:
            no_grad_set = set()

        if not type(output_names) is list:
            output_names = [output_names]

        numeric_grads = [
            get_numeric_gradient(
                self.scope,
                self.op,
                self.inputs,
                input_to_check,
                output_names,
                in_place=in_place) for input_to_check in inputs_to_check
        ]
        grad_names = [
            grad_var_name(input_to_check) for input_to_check in inputs_to_check
        ]

        cpu_place = core.CPUPlace()
        cpu_analytic_grads = [
            get_gradient(self.scope, self.op, self.inputs, self.outputs,
                         grad_name, cpu_place, no_grad_set)
            for grad_name in grad_names
        ]

        self.__assert_is_close(numeric_grads, cpu_analytic_grads, grad_names,
                               max_relative_error,
                               "Gradient Check On %s" % str(cpu_place))

        if core.is_compile_gpu() and self.op.support_gpu():
            gpu_place = core.GPUPlace(0)
            gpu_analytic_grads = [
                get_gradient(self.scope, self.op, self.inputs, self.outputs,
                             grad_name, gpu_place, no_grad_set)
                for grad_name in grad_names
            ]

            self.__assert_is_close(numeric_grads, gpu_analytic_grads,
                                   grad_names, max_relative_error,
                                   "Gradient Check On %s" % str(gpu_place))

            for c_grad, g_grad, name in itertools.izip(
                    cpu_analytic_grads, gpu_analytic_grads, grad_names):
                self.assertTrue(
                    np.allclose(
                        c_grad, g_grad, atol=1e-4),
                    "output name: " + name + " has diff")
