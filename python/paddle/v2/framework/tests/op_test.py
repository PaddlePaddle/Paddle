import unittest
import numpy as np
import random
import itertools
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


def grad_var_name(var_name):
    return var_name + "@GRAD"


def create_op(scope, op_type, inputs, outputs, attrs):
    kwargs = dict()

    def __create_var__(name, var_name):
        scope.new_var(var_name)
        kwargs[name].append(var_name)

    for in_name, in_dup in Operator.get_op_inputs(op_type):
        if in_name in inputs:
            kwargs[in_name] = []
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, _ in sub_in:
                    __create_var__(in_name, sub_in_name)
            else:
                __create_var__(in_name, in_name)

    for out_name, out_dup in Operator.get_op_outputs(op_type):
        if out_name in outputs:
            kwargs[out_name] = []
            if out_dup:
                sub_out = outputs[out_name]
                for sub_out_name, _ in sub_out:
                    __create_var__(out_name, sub_out_name)
            else:
                __create_var__(out_name, out_name)

    for attr_name in Operator.get_op_attr_names(op_type):
        if attr_name in attrs:
            kwargs[attr_name] = attrs[attr_name]

    return Operator(op_type, **kwargs)


def set_input(scope, op, inputs, place):
    def __set_input__(var_name, var):
        if isinstance(var, tuple) or isinstance(var, np.ndarray):
            tensor = scope.find_var(var_name).get_tensor()
            if isinstance(var, tuple):
                tensor.set_lod(var[1])
                var = var[0]
            tensor.set_dims(var.shape)
            tensor.set(var, place)
        elif isinstance(var, float):
            scope.find_var(var_name).set_float(var)
        elif isinstance(var, int):
            scope.find_var(var_name).set_int(var)

    for in_name, in_dup in Operator.get_op_inputs(op.type()):
        if in_name in inputs:
            if in_dup:
                sub_in = inputs[in_name]
                for sub_in_name, sub_in_val in sub_in:
                    __set_input__(sub_in_name, sub_in_val)
            else:
                __set_input__(in_name, inputs[in_name])


def set_output_grad(scope, op, outputs, place):
    def __set_tensor__(name):
        out_tensor = scope.find_var(name).get_tensor()
        grad_tensor = scope.new_var(grad_var_name(name)).get_tensor()
        out_dtype = out_tensor.dtype()
        if out_dtype == core.DataType.FP64:
            data = np.ones(out_tensor.shape(), dtype=np.float64)
        elif out_dtype == core.DataType.FP32:
            data = np.ones(out_tensor.shape(), dtype=np.float32)
        else:
            raise ValueError("Not supported data type " + str(out_dtype))

        grad_tensor.set(data, place)

    for out_name, out_dup in Operator.get_op_outputs(op.type()):
        if out_name in outputs:
            if out_dup:
                sub_out = outputs[out_name]
                for sub_out_name, _ in sub_out:
                    __set_tensor__(sub_out_name)
            else:
                __set_tensor__(out_name)


def get_numeric_gradient(scope,
                         op,
                         inputs,
                         input_to_check,
                         output_names,
                         delta=0.005,
                         in_place=False):
    set_input(scope, op, inputs, core.CPUPlace())

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
    tensor_to_check_dtype = tensor_to_check.dtype()
    if tensor_to_check_dtype == core.DataType.FP32:
        tensor_to_check_dtype = np.float32
    elif tensor_to_check_dtype == core.DataType.FP64:
        tensor_to_check_dtype = np.float64
    else:
        raise ValueError("Not supported data type " + str(
            tensor_to_check_dtype))

    gradient_flat = np.zeros(shape=(tensor_size, ), dtype=tensor_to_check_dtype)

    def __get_elem__(tensor, i):
        if tensor_to_check_dtype == np.float32:
            return tensor.get_float_element(i)
        else:
            return tensor.get_double_element(i)

    def __set_elem__(tensor, i, e):
        if tensor_to_check_dtype == np.float32:
            tensor.set_float_element(i, e)
        else:
            tensor.set_double_element(i, e)

    # we only compute gradient of one element each time.
    # we use a for loop to compute the gradient of every element.
    for i in xrange(tensor_size):
        if in_place:
            set_input(scope, op, inputs, core.CPUPlace())

        # get one input element throw it's index i.
        origin = __get_elem__(tensor_to_check, i)
        # add delta to it, run op and then get the sum of the result tensor.
        x_pos = origin + delta
        __set_elem__(tensor_to_check, i, x_pos)
        y_pos = get_output()

        if in_place:
            set_input(scope, op, inputs, core.CPUPlace())

        x_neg = origin - delta
        __set_elem__(tensor_to_check, i, x_neg)
        y_neg = get_output()

        __set_elem__(tensor_to_check, i, origin)
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

    op.run(scope, ctx)

    if no_grad_set is None:
        no_grad_set = set()

    backward_op = get_backward_op(scope, op, no_grad_set)
    set_output_grad(scope, op, outputs, place)

    backward_op.run(scope, ctx)

    out = np.array(scope.find_var(grad_name).get_tensor())
    return out


class OpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''Fix random seeds to remove randomness from tests'''
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        np.random.seed(123)
        random.seed(124)

    @classmethod
    def tearDownClass(cls):
        '''Restore random seeds'''
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    def check_output_with_place(self, place, atol):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()
        self.op = create_op(self.scope, self.op_type, op_inputs, op_outputs,
                            op_attrs)
        if isinstance(place, core.GPUPlace) and not self.op.support_gpu():
            return
        set_input(self.scope, self.op, self.inputs, place)
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
                            actual, expect, atol=atol),
                        "output name: " + out_name + " has diff.")
            else:
                actual = np.array(self.scope.find_var(out_name).get_tensor())
                expect = self.outputs[out_name]

                self.assertTrue(
                    np.allclose(
                        actual, expect, atol=atol),
                    "output name: " + out_name + " has diff.")

    def check_output(self, atol=1e-5):
        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))
        for place in places:
            self.check_output_with_place(place, atol)

    def __assert_is_close(self, numeric_grads, analytic_grads, names,
                          max_relative_error, msg_prefix):

        for a, b, name in itertools.izip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return ("%s Variable %s max gradient diff %f over limit %f, "
                        "the first error element is %d") % (
                            msg_prefix, name, max_diff, max_relative_error,
                            offset)

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
