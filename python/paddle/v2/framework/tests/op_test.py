import unittest
import numpy as np
import itertools
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


# TODO(pkuyym) simplify the code and add more comments
class TestUtils(object):
    @classmethod
    def grad_var_name(cls, var_name):
        return var_name + "@GRAD"

    @classmethod
    def get_tensor_by_name(cls, scope, var_name):
        var = scope.find_var(var_name)
        return var.get_tensor()

    @classmethod
    def create_variable(cls, scope, var_name, value=None, place=None):
        var = scope.new_var(var_name)
        if value is not None:
            assert place is not None, \
                'Place must be specified if value provided'
            tensor = var.get_tensor()
            tensor.set_dims(value.shape)
            tensor.set(value, place)

    @classmethod
    def create_op(cls, scope, op_type, inputs, outputs, attrs):
        kwargs = dict()
        # parepare parameters for creating operator
        for in_var_name, is_dup in Operator.get_op_inputs(op_type):
            if in_var_name in inputs:
                kwargs[in_var_name] = []
                if is_dup:
                    sub_vars = inputs[in_var_name]
                    for sub_var_name, _ in sub_vars:
                        cls.create_variable(scope, sub_var_name)
                        kwargs[in_var_name].append(sub_var_name)
                else:
                    cls.create_variable(scope, in_var_name)
                    kwargs[in_var_name].append(in_var_name)

        for out_var_name, is_dup in Operator.get_op_outputs(op_type):
            if out_var_name in outputs:
                kwargs[out_var_name] = []
                if is_dup:
                    sub_vars = outputs[out_var_name]
                    for sub_var_name, _ in sub_vars:
                        cls.create_variable(scope, sub_var_name)
                        kwargs[out_var_name].append(sub_var_name)
                else:
                    cls.create_variable(scope, out_var_name)
                    kwargs[out_var_name].append(out_var_name)

        for attr_name in Operator.get_op_attr_names(op_type):
            if attr_name in attrs:
                kwargs[attr_name] = attrs[attr_name]

        return Operator(op_type, **kwargs)

    @classmethod
    def get_backward_op(cls, scope, op, no_grad_set):
        backward_op = core.Operator.backward(op, no_grad_set)
        for in_var_name in backward_op.input_vars():
            cls.create_variable(scope, in_var_name)
        for out_var_name in backward_op.output_vars():
            cls.create_variable(scope, out_var_name)
        return backward_op

    @classmethod
    def _feed_var(cls, scope, var_name, value, place):
        tensor = cls.get_tensor_by_name(scope, var_name)
        lod_info = None
        if isinstance(value, tuple):
            data = value[0]
            lod_info = value[1]
        else:
            data = value
        tensor.set_dims(data.shape)
        tensor.set(data, place)
        if lod_info is not None:
            tensor.set_lod(lod_info)

    @classmethod
    def feed_input(cls, scope, op, inputs, place):
        for in_var_name, is_dup in Operator.get_op_inputs(op.type()):
            if in_var_name in inputs:
                if is_dup:
                    sub_vars = inputs[in_var_name]
                    for sub_var_name, sub_var_val in sub_vars:
                        cls._feed_var(scope, sub_var_name, sub_var_val, place)
                else:
                    in_var_val = inputs[in_var_name]
                    cls._feed_var(scope, in_var_name, in_var_val, place)

    @classmethod
    def dim_to_size(cls, dim):
        return reduce(lambda a, b: a * b, dim, 1)

    @classmethod
    def get_numeric_gradient(cls,
                             scope,
                             op,
                             inputs,
                             input_to_check,
                             output_names,
                             delta=0.005,
                             in_place=False,
                             strict=False):
        # compute numeric gradients on CPU
        cpu_place = core.CPUPlace()
        cls.feed_input(scope, op, inputs, cpu_place)
        op.infer_shape(scope)
        ctx = core.DeviceContext.create(cpu_place)

        tensor_to_check = cls.get_tensor_by_name(scope, input_to_check)
        tensor_size = cls.dim_to_size(tensor_to_check.get_dims())
        x_pos_jacobian = np.zeros((tensor_size, 0), dtype=np.float64)
        x_neg_jacobian = np.zeros((tensor_size, 0), dtype=np.float64)

        def concat_flatten_output(row, jacobian_matrix):
            op.run(scope, ctx)
            if jacobian_matrix.shape[1] == 0:
                # first time, concate output dynamically
                output_vals = []
                for output_name in output_names:
                    output_val = np.array(
                        cls.get_tensor_by_name(scope, output_name)).flatten()
                    output_vals = np.append(output_vals, output_val)
                # get dimension info, allocate memory
                jacobian_matrix.resize(
                    (tensor_size, len(output_vals)), refcheck=False)
                jacobian_matrix[row, :] = output_vals.flatten()
            else:
                start_idx = 0
                for output_name in output_names:
                    output_val = np.array(
                        cls.get_tensor_by_name(scope, output_name)).flatten()
                    jacobian_matrix[row, start_idx:start_idx+len(output_val)] \
                            = output_val
                    start_idx += len(output_val)

        for i in xrange(tensor_size):
            if in_place:
                cls.feed_input(scope, op, inputs, cpu_place)
            origin_val = tensor_to_check.get_float_element(i)
            x_pos = origin_val + delta
            tensor_to_check.set_float_element(i, x_pos)
            concat_flatten_output(i, x_pos_jacobian)
            if in_place:
                cls.feed_input(scope, op, inputs, cpu_place)
            x_neg = origin_val - delta
            tensor_to_check.set_float_element(i, x_neg)
            concat_flatten_output(i, x_neg_jacobian)
            tensor_to_check.set_float_element(i, origin_val)

        grad_jacobian = (x_pos_jacobian - x_neg_jacobian) / delta / 2
        # return numeric gradient jacobian matrix
        if strict == False:
            return grad_jacobian.sum(axis=1).reshape(tensor_to_check.shape())
        return grad_jacobian

    # TODO(pkuyym) should pass output_names not outputs
    @classmethod
    def get_simple_analytic_grads(cls,
                                  scope,
                                  op,
                                  inputs,
                                  outputs,
                                  grad_name,
                                  place,
                                  no_grad_set=None):
        ctx = core.DeviceContext.create(place)
        cls.feed_input(scope, op, inputs, place)
        # run forward
        op.infer_shape(scope)
        op.run(scope, ctx)

        if no_grad_set is None:
            no_grad_set = set()

        backward_op = cls.get_backward_op(scope, op, no_grad_set)
        # feed Input(Out@Grad), just set to one for all values
        for out_var_name, is_dup in Operator.get_op_outputs(op.type()):
            if out_var_name in outputs:
                if is_dup:
                    sub_vars = outputs[out_var_name]
                    for sub_var_name, _ in sub_vars:
                        out_var_tensor = cls.get_tensor_by_name(scope,
                                                                sub_var_name)
                        data = np.ones(out_var_tensor.shape(), dtype=np.float64)
                        cls.create_variable(
                            scope,
                            cls.grad_var_name(sub_var_name),
                            value=data,
                            place=place)
                else:
                    out_var_tensor = cls.get_tensor_by_name(scope, out_var_name)
                    data = np.ones(out_var_tensor.shape(), np.float64)
                    cls.create_variable(
                        scope,
                        cls.grad_var_name(out_var_name),
                        value=data,
                        place=place)

        backward_op.infer_shape(scope)
        backward_op.run(scope, ctx)
        out = np.array(cls.get_tensor_by_name(scope, grad_name))
        return out

    @classmethod
    def get_out_var_shapes(cls, scope, op, outputs):
        out_var_shapes = []
        for out_var_name, is_dup in Operator.get_op_outputs(op.type()):
            if out_var_name in outputs:
                if is_dup:
                    sub_vars = outputs[out_var_name]
                    for sub_var_name, _ in sub_vars:
                        out_var_tensor = cls.get_tensor_by_name(scope,
                                                                sub_var_name)
                        out_var_shapes.append(
                            (sub_var_name, out_var_tensor.shape()))
                else:
                    out_var_tensor = cls.get_tensor_by_name(scope, out_var_name)
                    out_var_shapes.append(
                        (out_var_name, out_var_tensor.shape()))
        return out_var_shapes

    # TODO(pkuyym) should pass output_names not outputs
    @classmethod
    def get_jacobian_analytic_grads(cls,
                                    scope,
                                    op,
                                    inputs,
                                    outputs,
                                    grad_name,
                                    place,
                                    no_grad_set=None):
        # only run forward one time
        ctx = core.DeviceContext.create(place)
        cls.feed_input(scope, op, inputs, place)
        op.infer_shape(scope)
        op.run(scope, ctx)

        # get shape for each outputs, may pass by outside
        out_var_shapes = cls.get_out_var_shapes(scope, op, outputs)
        accum_size = np.zeros((len(out_var_shapes)), dtype=np.int32)
        var_shape_idx = {}
        for i in xrange(len(out_var_shapes)):
            accum_size[i] = cls.dim_to_size(out_var_shapes[i][1]) + \
                    (accum_size[i - 1] if i > 0 else 0)
            var_shape_idx[out_var_shapes[i][0]] = i

        out_grad_values = np.zeros(accum_size[-1], dtype=np.float64)
        x_grad_jacobian = None

        backward_op = cls.get_backward_op(scope, op, no_grad_set)

        def fill_tensor(tensor_name, tensor, place):
            tensor_shape = tensor.shape()
            if tensor_name in var_shape_idx:
                idx = var_shape_idx[tensor_name]
                start = accum_size[idx - 1] if idx > 0 else 0
                data = out_grad_values[start:accum_size[idx]].reshape(
                    tensor_shape)
            else:
                data = np.zeros(tensor_shape, dtype=np.float64)
            tensor.set(data, place)

        for i in xrange(accum_size[-1]):
            # each time set 1 to one value
            out_grad_values[i] = 1
            # feed Input(Out@Grad)
            for out_var_name, is_dup in Operator.get_op_outputs(op.type()):
                if out_var_name in outputs:
                    if is_dup:
                        sub_vars = outputs[out_var_name]
                        for sub_var_name, _ in sub_vars:
                            out_var_tensor = cls.get_tensor_by_name(
                                scope, sub_var_name)
                            cls.create_variable(scope,
                                                cls.grad_var_name(sub_var_name))
                            out_grad_tensor = cls.get_tensor_by_name(
                                scope, cls.grad_var_name(sub_var_name))
                            out_grad_tensor.set_dims(out_var_tensor.shape())
                            fill_tensor(var_name, out_grad_tensor, place)
                    else:
                        out_var_tensor = cls.get_tensor_by_name(scope,
                                                                out_var_name)
                        cls.create_variable(scope,
                                            cls.grad_var_name(out_var_name))
                        out_grad_tensor = cls.get_tensor_by_name(
                            scope, cls.grad_var_name(out_var_name))
                        out_grad_tensor.set_dims(out_var_tensor.shape())
                        fill_tensor(out_var_name, out_grad_tensor, place)

            if no_grad_set is None:
                no_grad_set = set()

            backward_op.infer_shape(scope)
            backward_op.run(scope, ctx)
            # fill input gradient jacobian matrix
            x_grad_col = np.array(cls.get_tensor_by_name(scope, grad_name))
            if x_grad_jacobian is None:
                # get shape info, allocat memory
                x_grad_jacobian = np.zeros((x_grad_col.size, accum_size[-1]))
            x_grad_jacobian[:, i] = x_grad_col.flatten()
            # reset to zero
            out_grad_values[i] = 0

        return x_grad_jacobian


class OpTest(unittest.TestCase):
    def check_output_with_place(self, place):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()
        self.op = TestUtils.create_op(self.scope, self.op_type, op_inputs,
                                      op_outputs, op_attrs)

        if isinstance(place, core.GPUPlace) and not self.op.support_gpu():
            return

        TestUtils.feed_input(self.scope, self.op, self.inputs, place)
        self.op.infer_shape(self.scope)
        ctx = core.DeviceContext.create(place)
        self.op.run(self.scope, ctx)

        for out_var_name, is_dup in Operator.get_op_outputs(self.op.type()):
            if out_var_name not in self.outputs:
                continue

            if is_dup:
                sub_vars = self.outputs[out_var_name]
                if not isinstance(sub_vars, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))

                for sub_var_name, expect_val in sub_vars:
                    actual_val = np.array(
                        TestUtils.get_tensor_by_name(self.scope, sub_var_name))

                    self.assertTrue(
                        np.allclose(
                            actual_val, expect_val, atol=1e-05),
                        "output name: " + out_var_name + " has diff")
            else:
                actual_val = np.array(
                    TestUtils.get_tensor_by_name(self.scope, out_var_name))
                expect_val = self.outputs[out_var_name]
                self.assertTrue(
                    np.allclose(
                        actual_val, expect_val, atol=1e-05),
                    "output name: " + out_var_name + " has diff")

    def check_output(self):
        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))
        for place in places:
            self.check_output_with_place(place)

    def _assert_is_close(self, numeric_grads, analytic_grads, names,
                         max_relative_error, msg_prefix):
        for a, b, name in itertools.izip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1
            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return "%s Variable %s max gradient diff %f over limit %f, "\
                        "the first error element is %d" % (
                   msg_prefix, name, max_diff, max_relative_error, offset)

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   in_place=False,
                   max_relative_error=0.005,
                   strict=False):
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else dict()
        op_outputs = self.outputs if hasattr(self, "outputs") else dict()
        op_attrs = self.attrs if hasattr(self, "attrs") else dict()
        self.op = TestUtils.create_op(self.scope, self.op_type, op_inputs,
                                      op_outputs, op_attrs)

        if no_grad_set is None:
            no_grad_set = set()

        if not type(output_names) is list:
            output_names = [output_names]

        involved_outputs = {}
        for key, val in self.outputs.items():
            if key in output_names:
                involved_outputs[key] = val
            elif isinstance(val, list):
                sub_outs = []
                for sub_var_name, sub_var_val in val:
                    if sub_var_name in output_names:
                        sub_outs.append((sub_var_name, sub_var_val))
                involved_outputs[key] = sub_outs

        numeric_grads = [
            TestUtils.get_numeric_gradient(
                self.scope,
                self.op,
                self.inputs,
                input_to_check,
                output_names,
                in_place=in_place,
                strict=strict) for input_to_check in inputs_to_check
        ]

        grad_names = [
            TestUtils.grad_var_name(input_to_check) \
                    for input_to_check in inputs_to_check
        ]

        cpu_place = core.CPUPlace()

        if strict == False:
            cpu_analytic_grads = [
                TestUtils.get_simple_analytic_grads(
                    self.scope, self.op, self.inputs,
                    involved_outputs, grad_name, cpu_place, no_grad_set)
                for grad_name in grad_names
            ]
        else:
            cpu_analytic_grads = [
                TestUtils.get_jacobian_analytic_grads(
                    self.scope, self.op, self.inputs,
                    involved_outputs, grad_name, cpu_place, no_grad_set)
                for grad_name in grad_names
            ]

        self._assert_is_close(numeric_grads, cpu_analytic_grads, grad_names,
                              max_relative_error,
                              "Gradient Check On %s" % str(cpu_place))

        if core.is_compile_gpu() and self.op.support_gpu():
            gpu_place = core.GPUPlace(0)
            if strict == False:
                gpu_analytic_grads = [
                    TestUtils.get_simple_analytic_grads(
                        self.scope, self.op, self.inputs, involved_outputs,
                        grad_name, gpu_place, no_grad_set)
                    for grad_name in grad_names
                ]
            else:
                gpu_analytic_grads = [
                    TestUtils.get_jacobian_analytic_grads(
                        self.scope, self.op, self.inputs, involved_outputs,
                        grad_name, gpu_place, no_grad_set)
                    for grad_name in grad_names
                ]

            self._assert_is_close(numeric_grads, gpu_analytic_grads, grad_names,
                                  max_relative_error,
                                  "Gradient Check On %s" % str(gpu_place))

            for c_grad, g_grad, name in itertools.izip(
                    cpu_analytic_grads, gpu_analytic_grads, grad_names):
                self.assertTrue(
                    np.allclose(
                        c_grad, g_grad, atol=1e-4),
                    "output name: " + name + " has diff")
