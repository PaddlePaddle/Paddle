import unittest

import numpy
import itertools
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator

__all__ = ['get_numeric_gradient']


def create_op(op_type):
    # TODO need to set attrs
    kwargs = dict()
    for in_name in Operator.get_op_input_names(op_type):
        kwargs[in_name] = in_name
    for out_name in Operator.get_op_output_names(op_type):
        kwargs[out_name] = out_name

    return Operator(op_type, **kwargs)


def grad_var_name(var_name):
    return var_name + "@GRAD"


def empty_var_name():
    return "@EMPTY@"


def get_numeric_gradient(op,
                         input_values,
                         output_name,
                         input_to_check,
                         delta=0.005,
                         local_scope=None,
                         in_place=False):
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
    opts = op.outputs()
    for key in opts:
        for output in opts[key]:
            if local_scope.find_var(output) is None:
                local_scope.new_var(output).get_tensor()
    op.infer_shape(local_scope)

    # allocate output memory
    for key in opts:
        for output in opts[key]:
            local_scope.find_var(output).get_tensor().alloc_float(core.CPUPlace(
            ))

    cpu_ctx = core.DeviceContext.create(core.CPUPlace())

    def get_output():
        op.run(local_scope, cpu_ctx)
        return numpy.array(local_scope.find_var(output_name).get_tensor()).sum()

    def product(dim):
        return reduce(lambda a, b: a * b, dim, 1)

    def restore_inputs():
        for var_name in input_values:
            tensor_ = local_scope.find_var(var_name).get_tensor()
            tensor_.set(numpy.copy(input_values[var_name]), core.CPUPlace())

    # get the input tensor that we want to get it's numeric gradient.
    tensor_to_check = local_scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.get_dims())
    # prepare a numpy array to store the gradient.
    gradient_flat = numpy.zeros(shape=(tensor_size, ), dtype='float32')

    # we only compute gradient of one element each time.
    # we use a for loop to compute the gradient of every element.
    for i in xrange(tensor_size):
        if in_place:
            restore_inputs()
        # get one input element throw it's index i.
        origin = tensor_to_check.get_float_element(i)

        # add delta to it, run op and then get the sum of the result tensor.
        x_pos = origin + delta
        tensor_to_check.set_float_element(i, x_pos)
        y_pos = get_output()

        # plus delta to this element, run op and get the sum of the result tensor.
        if in_place:
            restore_inputs()
        x_neg = origin - delta
        tensor_to_check.set_float_element(i, x_neg)
        y_neg = get_output()

        # restore old value
        tensor_to_check.set_float_element(i, origin)

        # compute the gradient of this element and store it into a numpy array.
        gradient_flat[i] = (y_pos - y_neg) / delta / 2

    # reshape the gradient result to the shape of the source tensor.
    return gradient_flat.reshape(tensor_to_check.get_dims())


class GradientChecker(unittest.TestCase):
    def __get_gradient(self, forward_op, backward_op, input_value, grad_names,
                       place):
        """Get the input gradients after running forward and backward operators
        on the given places.

        :param forward_op: forward operator
        :type forward_op: Operator
        :param backward_op: backward operator
        :type backward_op: Operator
        :param input_value: input values.
        :type input_value: dict{string:numpy.array}
        :param grad_names: the names of returned input gradients.
        :type input_value: a list of string
        :param place: the device type.
        :type place: CPUPlace or GPUPlace
        :return: the input grdients of given grad_names.
        :rtype: a list of numpy.array
        """
        scope = core.Scope()
        ctx = core.DeviceContext.create(place)

        inputs = forward_op.inputs()
        in_names = [item for k in inputs for item in inputs[k]]
        outputs = forward_op.outputs()
        out_names = [item for k in outputs for item in outputs[k]]

        # create input var and set value
        for name, value in input_value.iteritems():
            if name not in in_names:
                raise ValueError(name + "does not exist in Op's inputs.")
            var = scope.new_var(name).get_tensor()
            var.set_dims(value.shape)
            var.set(value, place)

        # run forward op
        for out_name in out_names:
            scope.new_var(out_name)
        forward_op.infer_shape(scope)
        forward_op.run(scope, ctx)

        # set output var's shape
        # set output grad to ones
        for name in out_names:
            out_tensor = scope.find_var(name).get_tensor()
            grad_tensor = scope.new_var(grad_var_name(name)).get_tensor()
            grad_tensor.set_dims(out_tensor.shape())
            data = numpy.ones(out_tensor.shape(), dtype=numpy.float32)
            grad_tensor.set(data, place)

        # run backward op
        backward_outs = backward_op.outputs()
        backward_names = [
            item for key in backward_outs for item in backward_outs[key]
        ]
        for name in backward_names:
            scope.new_var(name)

        backward_op.infer_shape(scope)
        backward_op.run(scope, ctx)

        outs = [
            numpy.array(scope.find_var(name).get_tensor())
            for name in grad_names
        ]
        return outs

    def compare_grad(self, forward_op, input_value, no_grad_set=None):
        """ Compare the input gradients between CPU and GPU for the given forward
        operator.

        :param forward_op: forward operator
        :type forward_op: Operator
        :param input_value: input values.
        :type input_value: dict{string:numpy.array}
        :param no_grad_set: the set of variables names without gradients.
        :type no_grad_set: a set of string
        :raises: AssertionError, there is different gradient value.
        """
        if no_grad_set is None:
            no_grad_set = set()
        backward_op = core.Operator.backward(forward_op, no_grad_set)
        # return if not compile with GPU or not implementing GPU kernel
        if not (core.is_compile_gpu() and backward_op.support_gpu()):
            return

        outputs = backward_op.outputs()
        out_names = [item for k in outputs for item in outputs[k]]
        out_names = filter(lambda x: x != empty_var_name(), out_names)
        cpu_grads = self.__get_gradient(forward_op, backward_op, input_value,
                                        out_names, core.CPUPlace())
        gpu_grads = self.__get_gradient(forward_op, backward_op, input_value,
                                        out_names, core.GPUPlace(0))

        for c_grad, g_grad, name in itertools.izip(cpu_grads, gpu_grads,
                                                   out_names):
            self.assertTrue(
                numpy.allclose(
                    c_grad, g_grad, atol=1e-4),
                "output name: " + name + " has diff")

    def __assert_is_close(self, numeric_grads, analytic_grads, names,
                          max_relative_error, msg_prefix):
        """Use relative error for the comparison.

        :param numeric_grads: the numerical graidents.
        :type numeric_grads: a list of numpy.array
        :param analytic_grads: the analytical graidents.
        :type analytic_grads: a list of numpy.array
        :param name: the names of gradients, used to print for debug.
        :type names: a list of string
        :param msg_prefix: string info, used to print for debug.
        :type msf_prefix: string
        """
        for a, b, name in itertools.izip(numeric_grads, analytic_grads, names):
            print "a=%s ; b=%s" % (a, b)
            abs_a = numpy.abs(a)
            # if abs_a is nearly zero, then use abs error for a, not relative
            # error.
            abs_a[abs_a < 1e-3] = 1

            diff_mat = numpy.abs(a - b) / abs_a
            max_diff = numpy.max(diff_mat)

            def err_msg():
                offset = numpy.argmax(diff_mat > max_relative_error)
                return "%s Variable %s max gradient diff %f over limit %f, the first " \
                       "error element is %d" % (
                       msg_prefix, name, max_diff, max_relative_error, offset)

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def check_grad(self,
                   forward_op,
                   input_vars,
                   inputs_to_check,
                   output_name,
                   no_grad_set=None,
                   only_cpu=False,
                   in_place=False,
                   max_relative_error=0.005):
        """
        :param forward_op: used to create backward_op
        :param input_vars: numpy value of input variable. The following
            computation will use these variables.
        :param inputs_to_check: inputs var names that should check gradient.
        :param output_name: the output variable name of forward network.
        :param max_relative_error: The relative tolerance parameter.
        :param no_grad_set: used when create backward ops
        :param only_cpu: only compute and check gradient on cpu kernel.
        :return:
        """
        if no_grad_set is None:
            no_grad_set = set()

        no_tmp_out = forward_op.no_intermediate_outputs()
        if len(no_tmp_out) != 1:
            raise ValueError("non temp out_names should be 1")

        inputs = forward_op.inputs()
        in_names = [item for k in inputs for item in inputs[k]]
        for no_grad in no_grad_set:
            if no_grad not in in_names:
                raise ValueError("no_grad should be in in_names")
            if no_grad in inputs_to_check:
                raise ValueError("no_grad should not be in inputs_to_check")

        backward_op = core.Operator.backward(forward_op, no_grad_set)

        places = [core.CPUPlace()]
        if not only_cpu and core.is_compile_gpu() and backward_op.support_gpu():
            places.append(core.GPUPlace(0))

        # get numerical gradients
        numeric_grads = [
            get_numeric_gradient(
                forward_op, input_vars, output_name, name, in_place=in_place)
            for name in inputs_to_check
        ]

        check_names = [grad_var_name(name) for name in inputs_to_check]
        for place in places:
            analytic_grads = self.__get_gradient(forward_op, backward_op,
                                                 input_vars, check_names, place)
            self.__assert_is_close(numeric_grads, analytic_grads, check_names,
                                   max_relative_error,
                                   "Gradient Check On %s" % str(place))
