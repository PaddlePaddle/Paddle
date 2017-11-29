import unittest
import numpy as np
import random
import itertools
import paddle.v2.fluid.core as core
import collections
from paddle.v2.fluid.backward import append_backward_ops
from paddle.v2.fluid.op import Operator
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.framework import Program, OpProtoHolder


def randomize_probability(batch_size, class_num, dtype='float32'):
    prob = np.random.uniform(
        0.1, 1.0, size=(batch_size, class_num)).astype(dtype)
    prob_sum = prob.sum(axis=1)
    for i in xrange(len(prob)):
        prob[i] /= prob_sum[i]
    return prob


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

    def _init_program(self):
        """Initialize test environment

        Initialize `Program` and  global `Block`.
        """
        self.op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)
        self.program = Program()
        self.block = self.program.global_block()
        # If operator performs in place computation on a variable, in_place_map 
        # maps both input and output to the same variable .
        self.cached_var = {}
        if not hasattr(self, "in_place_map"):
            self.in_place_map = {}
        # Initialize attributes
        if not hasattr(self, "inputs"):
            self.inputs = {}
        if not hasattr(self, "outputs"):
            self.outputs = {}
        if not hasattr(self, "attrs"):
            self.attrs = {}

    def _init_var_desc(self, name, feed_value=None, dtype="float32"):
        """Initialize description for one varible in current block.

        Create a variable description in current block if it dosen't exist.

        :param name: variable name. 
        :type name: basestring.
        :param feed_value: tensor value and shape that used to initialize the variable. 
        :type feed_value: 1) numpy `array`, in which case lod_level is 0; 
                          2) 2d tuple, where the first element is numpy `array` and 
                          the second element is a `list` specifying the lod_level.
        :param dtype: data type of tensor. 
        :type dtype: basestring or one of numpy data types.
        :return: variable in current block.
        :rtype: Variable.
        """
        block = self.block
        if block.has_var(name):
            return block.var(name)

        shape = None
        lod_level = None
        if feed_value is not None:
            if isinstance(feed_value, tuple):
                shape = list(feed_value[0].shape)
                lod_level = len(feed_value[1])
            else:
                shape = list(feed_value.shape)
                lod_level = 0
        return block.create_var(
            dtype=dtype, shape=shape, lod_level=lod_level, name=name)

    def _init_var_descs(self, var_protos, feed_values, is_input):
        """Initialize descriptions for varibles in current block.

        Create variable description for input variables or output variables of an operator.

        :param var_protos: input or output variable protobuf message. 
        :type name: list of VarProto.
        :param feed_values: a dictionary that maps variable name to its initial values .
        :type feed_values: dict.
        :param is_input: True if var_protos contains input variables .
        :type is_input: Boolean. 
        :return: a dictionary that maps variable names to the variable description object. 
        :rtype: dict.
        """
        var_dict = {}

        def value_data_type(value):
            if value is None:
                return np.float32
            if isinstance(value, tuple):
                value = value[0]
            return value.dtype

        for var_proto in var_protos:
            var_name = str(
                var_proto.
                name)  # name is a unicode object. but this shouldn't matter.
            if var_name not in feed_values:
                if var_proto.dispensable and is_input:
                    continue
                assert var_proto.dispensable or var_proto.intermediate, "Missing {}".format(
                    var_name)

            # if it is inplace computation, use the same variable
            if var_name in self.in_place_map:
                linked_var = self.in_place_map[var_name]
                if linked_var in self.cached_var:
                    var_dict[var_name] = self.cached_var[linked_var]
                    continue
                var_name = linked_var

            if var_proto.duplicable:
                assert isinstance(feed_values[var_name], list), \
                    "Duplicable {} should be set as list".format(var_name)
                var_dict[var_name] = [
                    self._init_var_desc(
                        name, feed_value=value, dtype=value_data_type(value))
                    for name, value in feed_values[var_name]
                ]
            else:
                value = feed_values.get(var_name, None)
                var_dict[var_name] = self._init_var_desc(
                    var_name, feed_value=value, dtype=value_data_type(value))
        self.cached_var.update(var_dict)
        return var_dict

    @staticmethod
    def _feed_vars(var_descs, var_values, place):
        """Feed values to tensors in variable

        :param var_descs: a list of variable descriptions 
        :type name: list.
        :param var_values: a dictionary that maps variable name to its initial values.
        :type var_values: dict.
        :param place: Gpu or Cpu place .
        :type is_input: Place. 
        :return: a dictionary that maps variable names to its corresponding tensor. 
        :rtype: dict.
        """
        feed_map = {}
        assert all([key in var_values for key in var_descs.keys()
                    ]), "Not all variable are fed with values"
        for var_name, var_desc in var_descs.iteritems():
            fed_value = var_values[var_name]
            if isinstance(var_desc, list):
                for name, value in fed_value:
                    tensor = core.LoDTensor()
                    if isinstance(value, tuple):
                        tensor.set(value[0], place)
                        tensor.set_lod(value[1])
                    else:
                        tensor.set(value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(fed_value, tuple):
                    tensor.set(fed_value[0], place)
                    tensor.set_lod(fed_value[1])
                else:
                    tensor.set(fed_value, place)
                feed_map[var_name] = tensor

        return feed_map

    def _compile_op(self, place):
        """create an operator in current block.

        :param place: Gpu or Cpu place. 
        :type name: Place.
        :return: 3d tuple (operator, input variable descriptions, 
                 output variable descriptions) 
        :rtype: tuple.
        """
        # compile time prepare
        block = self.block
        input_var_descs = self._init_var_descs(
            self.op_proto.inputs, feed_values=self.inputs, is_input=True)
        output_var_descs = self._init_var_descs(
            self.op_proto.outputs, feed_values=self.outputs, is_input=False)
        op = block.append_op(
            type=self.op_type,
            inputs=input_var_descs,
            outputs=output_var_descs,
            attrs=self.attrs if hasattr(self, "attrs") else dict())
        # infer variable type and infer shape in compile-time
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        return op, input_var_descs, output_var_descs

    @staticmethod
    def _get_fetch_list(var_descs, place, filter=None):
        """Get a list of variable descriptions that needs to be fetched from the result.

        :param var_descs: variable descriptions. 
        :type var_descs: list.
        :param place: Gpu or Cpu place. 
        :type place: Place.
        :param filter: If provided, only variable names in filter is added into fetch list. 
        :type filter: list.
        :return: a list of variable descriptions needs to be fetched. 
        :rtype: list.
        """
        fetch_list = []
        # flat var_descs values
        for var_name, var_desc in var_descs.iteritems():
            sub_vars = []
            if isinstance(var_desc, list):
                sub_filters = [sub_var.name for sub_var in var_desc]
            if isinstance(var_desc, list):
                if filter is not None:
                    if var_name not in filter:
                        var_desc = [
                            sub_var for sub_var in var_desc
                            if sub_var.name in filter
                        ]
                fetch_list.extend(var_desc)
            else:
                if filter is not None and var_name not in filter:
                    continue
                fetch_list.append(var_desc)

        return fetch_list

    def _execute(self, feed_map, fetch_list, place):
        """Run the operator with given inputs and outputs.

        :param feed_map: input tensors.
        :type feed_map: dict.
        :param fetch_list: output variables.
        :type fetch_list: list.
        :param place: Gpu or Cpu place. 
        :type place: Place.
        :return: output values.
        :rtype: list.
        """
        exe = Executor(place)
        return exe.run(self.program,
                       feed=feed_map,
                       fetch_list=fetch_list,
                       return_numpy=False)

    def _compare_results(self, place, outs, fetch_list, atol):
        """Check differences between outputs

        :param place: Gpu or Cpu place. 
        :type place: Place.
        :param outs: output values.
        :type outs: list.
        :param fetch_list: output variables.
        :type fetch_list: list.
        :param atol: minimum difference allowed.
        :type atol: float.
        """

        def find_actual(target_name, fetch_list):
            if target_name in self.in_place_map:
                target_name = self.in_place_map[target_name]
            found = [
                i for i, var in enumerate(fetch_list) if var.name == target_name
            ]
            self.assertTrue(
                len(found) == 1, "Found {} {}".format(len(found), target_name))
            return found[0]

        for out_name, out_dup in Operator.get_op_outputs(self.op_type):
            if out_name not in self.outputs:
                continue

            if out_dup:
                sub_out = self.outputs[out_name]
                if not isinstance(sub_out, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))
                for sub_out_name, expect in sub_out:
                    idx = find_actual(sub_out_name, fetch_list)
                    actual = outs[idx]
                    actual_t = np.array(actual)
                    expect_t = expect[0] \
                        if isinstance(expect, tuple) else expect
                    self.assertTrue(
                        np.allclose(
                            actual_t, expect_t, atol=atol),
                        "Output (" + sub_out_name + ") has diff at " +
                        str(place))
                    if isinstance(expect, tuple):
                        self.assertListEqual(
                            actual.lod(), expect[1], "Output (" + sub_out_name +
                            ") has different lod at " + str(place))
            else:
                idx = find_actual(out_name, fetch_list)
                actual = outs[idx]
                actual_t = np.array(actual)
                expect = self.outputs[out_name]
                expect_t = expect[0] if isinstance(expect, tuple) else expect
                self.assertTrue(
                    np.allclose(
                        actual_t, expect_t, atol=atol),
                    "Output (" + out_name + ") has diff at " + str(place))
                if isinstance(expect, tuple):
                    self.assertListEqual(actual.lod(), expect[1],
                                         "Output (" + out_name +
                                         ") has different lod at " + str(place))

    def check_output_with_place(self, place, atol):
        """Check output on the specific place(CPU or GPU)

        :param place: Gpu or Cpu place. 
        :type place: Place.
        :param atol: minimum difference allowed.
        :type atol: float.
        """
        self._init_program()
        op, input_var_descs, output_var_descs = self._compile_op(place)
        feed_map = self._feed_vars(input_var_descs, self.inputs, place)
        fetch_list = self._get_fetch_list(
            output_var_descs, place, filter=self.outputs.keys())
        outs = self._execute(feed_map, fetch_list, place)
        self._compare_results(place, outs, fetch_list, atol)

    def check_output(self, atol=1e-5):
        """Check operator forward process

        :param atol: minimum difference allowed.
        :type atol: float.
        """
        places = [core.CPUPlace()]
        if core.is_compile_gpu() and core.op_support_gpu(self.op_type):
            places.append(core.GPUPlace(0))
        for place in places:
            self.check_output_with_place(place, atol)

    def __assert_is_close(self, numeric_grads, analytic_grads, names,
                          max_relative_error, msg_prefix):
        """Check differences between numerical gradient and analytical gradient

        :param numeric_grads: nemerical gradient.
        :type numeric_grads: numpy `array`.
        :param analytic_grads: analytical gradient.
        :type analytic_grads: numpy `array`.
        :param names: variable name.
        :type names: basestring.
        :param max_relative_error: minimum difference allowed.
        :type max_relative_error: float.
        :param msg_prefix: error message prefix.
        :type msg_prefix: basestring.
        """
        for a, b, name in itertools.izip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return ("%s Variable %s max gradient diff %f over limit %f, "
                        "the first error element is %d, %f, %f") % (
                            msg_prefix, name, max_diff, max_relative_error,
                            offset, a.flatten()[offset], b.flatten()[offset])

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def _numeric_gradient(self,
                          input_var_descs,
                          output_var_descs,
                          input_to_check,
                          output_names,
                          place,
                          delta=0.005,
                          in_place=False):
        """Compute numerical gradient for one input variable

        :param input_var_descs: input variable descriptions. 
        :type input_var_descs: list.
        :param output_var_descs: input variable descriptions. 
        :type output_var_descs: list.
        :param input_to_check: the input variable to compute gradient. 
        :type input_to_check: basestring.
        :param output_names: the output variable names. 
        :type output_names: list.
        :param place: Gpu or Cpu place. 
        :type place: Place.
        :param delta: the small value added to original value. 
        :type delta: float.
        :param in_place: wheter the operator performs in-place computation.
        :type in_place: boolean.
        :return: numerical gradients.
        :rtype: numpy `array`
        """
        ctx = core.DeviceContext.create(core.CPUPlace())

        def _init_input():
            feed_map = self._feed_vars(input_var_descs, self.inputs, place)
            tensor_to_check = feed_map.get(input_to_check, None)
            assert tensor_to_check is not None, "Can't find input name {}".format(
                input_to_check)
            return feed_map, tensor_to_check

        feed_map, tensor_to_check = _init_input()
        tensor_to_check_dtype = {
            core.DataType.FP32: np.float32,
            core.DataType.FP64: np.float64
        }.get(tensor_to_check.dtype(), None)
        if tensor_to_check_dtype is None:
            raise ValueError("Not supported data type " + str(
                tensor_to_check_dtype))
        tensor_size = np.prod(tensor_to_check.get_dims())
        gradient_flat = np.zeros(
            shape=(tensor_size, ), dtype=tensor_to_check_dtype)
        fetch_list = self._get_fetch_list(
            output_var_descs, place, filter=output_names)
        exe = Executor(place)

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

        def run_once():
            outs = exe.run(self.program,
                           feed=feed_map,
                           fetch_list=fetch_list,
                           return_numpy=False)
            return np.nanmean(
                np.array([np.array(out) for out in outs]).mean(axis=0))

        for i in xrange(tensor_size):
            if in_place:
                feed_map, tensor_to_check = _init_input()
            # positive
            origin = __get_elem__(tensor_to_check, i)
            __set_elem__(tensor_to_check, i, origin + delta)
            y_pos = run_once()

            if in_place:
                feed_map, tensor_to_check = _init_input()
            # negative     
            __set_elem__(tensor_to_check, i, origin - delta)
            y_neg = run_once()
            # restore            
            __set_elem__(tensor_to_check, i, origin)
            gradient_flat[i] = (y_pos - y_neg) / delta / 2.

        return gradient_flat.reshape(tensor_to_check.get_dims())

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=set(),
                   numeric_grad_delta=0.005,
                   in_place=False,
                   max_relative_error=0.005,
                   user_defined_grads=None):
        """Check the backward computation of operator 

        :param inputs_to_check: the input variables to compute gradient. 
        :type inputs_to_check: list.
        :param output_names: the output variable names. 
        :type output_names: list.
        :param no_grad_set: the set of variables that doesn't need to compute gradient
        :type no_grad_set: set
        :param place: Gpu or Cpu place. 
        :type place: Place.
        :param numeric_grad_delta: the small value added to original value 
        :type numeric_grad_delta: delta: float.
        :param in_place: wheter the operator performs in-place computation
        :type in_place: boolean.
        :param max_relative_error: minimum difference allowed.
        :type max_relative_error: float.
        :param user_defined_grads: if provided, use these user defined gradients as reference.
        :type user_defined_grads: list.
        """
        self._init_program()
        place = core.CPUPlace()
        op, input_var_descs, output_var_descs = self._compile_op(place)

        if not isinstance(output_names, list):
            output_names = [output_names]
        numeric_grads = user_defined_grads or [
            self._numeric_gradient(
                input_var_descs,
                output_var_descs,
                input_to_check,
                output_names,
                place,
                delta=numeric_grad_delta,
                in_place=in_place) for input_to_check in inputs_to_check
        ]

        cpu_analytic_grads = self._get_gradient(inputs_to_check, place,
                                                output_names, no_grad_set)

        self.__assert_is_close(numeric_grads, cpu_analytic_grads,
                               inputs_to_check, max_relative_error,
                               "Gradient Check On %s" % str(place))

        if core.is_compile_gpu() and core.op_support_gpu(self.op_type):
            gpu_place = core.GPUPlace(0)
            gpu_analytic_grads = self._get_gradient(inputs_to_check, gpu_place,
                                                    output_names, no_grad_set)

            self.__assert_is_close(numeric_grads, gpu_analytic_grads,
                                   inputs_to_check, max_relative_error,
                                   "Gradient Check On %s" % str(gpu_place))

    def _get_gradient(self, input_to_check, place, output_names, no_grad_set):
        """Compute analytical gradient for one input variable

        :param input_to_check: the input variable to compute gradient. 
        :type input_to_check: basestring.
        :param place: Gpu or Cpu place. 
        :type place: Place.
        :param output_names: the output variable names. 
        :type output_names: list.
        :param no_grad_set: the set of variables that doesn't need to compute gradient
        :type no_grad_set: set
        :return: numerical gradients.
        :rtype: numpy `array`
        """
        self._init_program()
        forward_op, input_var_descs, output_var_descs = self._compile_op(place)
        feed_map = self._feed_vars(input_var_descs, self.inputs, place)

        block = self.block
        mean_inputs = map(block.var, output_names)
        if len(mean_inputs) == 1:
            loss = block.create_var(dtype=mean_inputs[0].dtype, shape=[1])
            op = block.append_op(
                inputs={"X": mean_inputs}, outputs={"Out": loss}, type='mean')
            op.desc.infer_var_type(block.desc)
            op.desc.infer_shape(block.desc)
        else:
            avg_sum = []
            debug_idx = 0
            for cur_loss in mean_inputs:
                debug_idx += 1
                cur_avg_loss = block.create_var(
                    dtype=cur_loss.data_type, shape=[1])
                op = block.append_op(
                    inputs={"X": [cur_loss]},
                    outputs={"Out": [cur_avg_loss]},
                    type="mean")
                op.desc.infer_var_type(block.desc)
                op.desc.infer_shape(block.desc)
                avg_sum.append(cur_avg_loss)

            loss_sum = block.create_var(dtype=avg_sum[0].data_type, shape=[1])
            op_sum = block.append_op(
                inputs={"X": avg_sum}, outputs={"Out": loss_sum}, type='sum')
            op_sum.desc.infer_var_type(block.desc)
            op_sum.desc.infer_shape(block.desc)

            loss = block.create_var(dtype=loss_sum.data_type, shape=[1])
            op_loss = block.append_op(
                inputs={"X": loss_sum},
                outputs={"Out": loss},
                type='scale',
                attrs={'scale': 1.0 / float(len(avg_sum))})
            op_loss.desc.infer_var_type(block.desc)
            op_loss.desc.infer_shape(block.desc)

        param_grad_list = append_backward_ops(
            loss=loss, parameter_list=input_to_check, no_grad_set=no_grad_set)
        fetch_list = [g for _, g in param_grad_list]

        result = self._execute(feed_map, fetch_list, place)
        return map(np.array, result)
