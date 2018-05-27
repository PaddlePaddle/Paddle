#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import random
import itertools
import paddle.fluid.core as core
import collections
from paddle.fluid.backward import append_backward
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder


def randomize_probability(batch_size, class_num, dtype='float32'):
    prob = np.random.uniform(
        0.1, 1.0, size=(batch_size, class_num)).astype(dtype)
    prob_sum = prob.sum(axis=1)
    for i in xrange(len(prob)):
        prob[i] /= prob_sum[i]
    return prob


def create_op(scope, op_type, inputs, outputs, attrs):
    kwargs = dict()

    op_maker = core.op_proto_and_checker_maker
    op_role_attr_name = op_maker.kOpRoleAttrName()

    if op_role_attr_name not in attrs:
        attrs[op_role_attr_name] = int(op_maker.OpRole.Forward)

    def __create_var__(name, var_name):
        scope.var(var_name).get_tensor()
        kwargs[name].append(var_name)

    for in_name, in_dup in Operator.get_op_inputs(op_type):
        if in_name in inputs:
            kwargs[in_name] = []
            if in_dup:
                sub_in = inputs[in_name]
                for item in sub_in:
                    sub_in_name, _ = item[0], item[1]
                    __create_var__(in_name, sub_in_name)
            else:
                __create_var__(in_name, in_name)

    for out_name, out_dup in Operator.get_op_outputs(op_type):
        if out_name in outputs:
            kwargs[out_name] = []
            if out_dup:
                sub_out = outputs[out_name]
                for item in sub_out:
                    sub_out_name, _ = item[0], item[1]
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
                for item in sub_in:
                    sub_in_name, sub_in_val = item[0], item[1]
                    __set_input__(sub_in_name, sub_in_val)
            else:
                __set_input__(in_name, inputs[in_name])


def get_numeric_gradient(place,
                         scope,
                         op,
                         inputs,
                         input_to_check,
                         output_names,
                         delta=0.005,
                         in_place=False):
    # FIXME: change this method by compile time concepts
    set_input(scope, op, inputs, place)

    def product(dim):
        return reduce(lambda a, b: a * b, dim, 1)

    def get_output():
        sum = []
        for output_name in output_names:
            op.run(scope, place)
            sum.append(
                np.array(scope.find_var(output_name).get_tensor()).mean())
        return np.array(sum).mean()

    tensor_to_check = scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.get_dims())
    tensor_to_check_dtype = tensor_to_check.dtype()
    if tensor_to_check_dtype == core.VarDesc.VarType.FP32:
        tensor_to_check_dtype = np.float32
    elif tensor_to_check_dtype == core.VarDesc.VarType.FP64:
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
            set_input(scope, op, inputs, place)

        # get one input element throw it's index i.
        origin = __get_elem__(tensor_to_check, i)
        # add delta to it, run op and then get the sum of the result tensor.
        x_pos = origin + delta
        __set_elem__(tensor_to_check, i, x_pos)
        y_pos = get_output()

        if in_place:
            set_input(scope, op, inputs, place)

        x_neg = origin - delta
        __set_elem__(tensor_to_check, i, x_neg)
        y_neg = get_output()

        __set_elem__(tensor_to_check, i, origin)
        gradient_flat[i] = (y_pos - y_neg) / delta / 2

    return gradient_flat.reshape(tensor_to_check.get_dims())


def append_input_output(block, op_proto, np_list, is_input):
    '''Insert VarDesc and generate Python variable instance'''
    proto_list = op_proto.inputs if is_input else op_proto.outputs

    def create_var(block, name, np_list, var_proto):
        if name not in np_list:
            assert var_proto.intermediate, "{} not found".format(name)
            shape = None
            lod_level = None
        else:
            np_value = np_list[name]
            if isinstance(np_value, tuple):
                shape = list(np_value[0].shape)
                lod_level = len(np_value[1])
            else:
                shape = list(np_value.shape)
                lod_level = 0
        return block.create_var(
            dtype="float32", shape=shape, lod_level=lod_level, name=name)

    var_dict = {}
    for var_proto in proto_list:
        var_name = str(var_proto.name)
        if is_input:
            if (var_name not in np_list) and var_proto.dispensable:
                continue
            assert (var_name in np_list) or (var_proto.dispensable), \
                "Missing {} as input".format(var_name)
        if var_proto.duplicable:
            assert isinstance(np_list[var_name], list), \
                "Duplicable {} should be set as list".format(var_name)
            var_list = []
            for (name, np_value) in np_list[var_name]:
                var_list.append(
                    create_var(block, name, {name: np_value}, var_proto))
            var_dict[var_name] = var_list
        else:
            var_dict[var_name] = create_var(block, var_name, np_list, var_proto)

    return var_dict


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

    def feed_var(self, input_vars, place):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.LoDTensor()
                    if isinstance(np_value, tuple):
                        tensor.set(np_value[0], place)
                        tensor.set_lod(np_value[1])
                    else:
                        tensor.set(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(self.inputs[var_name], tuple):
                    tensor.set(self.inputs[var_name][0], place)
                    tensor.set_lod(self.inputs[var_name][1])
                else:
                    tensor.set(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def calc_output(self, place):
        outs, _ = self._calc_output(place)
        return outs

    def _calc_output(self, place):
        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

        program = Program()
        block = program.global_block()

        inputs = append_input_output(block, op_proto, self.inputs, True)
        outputs = append_input_output(block, op_proto, self.outputs, False)
        op = block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=self.attrs if hasattr(self, "attrs") else dict())
        # infer variable type and infer shape in compile-time
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        fetch_list = []
        for var_name, var in outputs.iteritems():
            if var_name in self.outputs:
                if isinstance(var, list):
                    for v in var:
                        fetch_list.append(v)
                else:
                    fetch_list.append(var)

        feed_map = self.feed_var(inputs, place)

        exe = Executor(place)
        outs = exe.run(program,
                       feed=feed_map,
                       fetch_list=fetch_list,
                       return_numpy=False)
        return outs, fetch_list

    def check_output_with_place(self, place, atol):
        outs, fetch_list = self._calc_output(place)
        for out_name, out_dup in Operator.get_op_outputs(self.op_type):
            if out_name not in self.outputs:
                continue

            def find_actual(target_name, fetch_list):
                found = [
                    i for i, var in enumerate(fetch_list)
                    if var.name == target_name
                ]
                self.assertTrue(
                    len(found) == 1, "Found {} {}".format(
                        len(found), target_name))
                return found[0]

            if out_dup:
                sub_out = self.outputs[out_name]
                if not isinstance(sub_out, list):
                    raise AssertionError("sub_out type %s is not list",
                                         type(sub_out))
                for item in sub_out:
                    sub_out_name, expect = item[0], item[1]
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
                    "Output (" + out_name + ") has diff at " + str(place) +
                    str(actual_t) + "\n" + str(expect_t))
                if isinstance(expect, tuple):
                    self.assertListEqual(actual.lod(), expect[1],
                                         "Output (" + out_name +
                                         ") has different lod at " + str(place))

    def check_output(self, atol=1e-5):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(self.op_type):
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_output_with_place(place, atol)

    def check_output_customized(self, checker):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(self.op_type):
            places.append(core.CUDAPlace(0))
        for place in places:
            outs = self.calc_output(place)
            outs = [np.array(out) for out in outs]
            checker(outs)

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
                        "the first error element is %d, %f, %f") % (
                            msg_prefix, name, max_diff, max_relative_error,
                            offset, a.flatten()[offset], b.flatten()[offset])

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def check_grad(self,
                   inputs_to_check,
                   output_names,
                   no_grad_set=None,
                   numeric_grad_delta=0.005,
                   in_place=False,
                   max_relative_error=0.005,
                   user_defined_grads=None):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(self.op_type):
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_grad_with_place(place, inputs_to_check, output_names,
                                       no_grad_set, numeric_grad_delta,
                                       in_place, max_relative_error,
                                       user_defined_grads)

    def check_grad_with_place(self,
                              place,
                              inputs_to_check,
                              output_names,
                              no_grad_set=None,
                              numeric_grad_delta=0.005,
                              in_place=False,
                              max_relative_error=0.005,
                              user_defined_grads=None):
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

        numeric_grads = user_defined_grads or [
            get_numeric_gradient(
                place,
                self.scope,
                self.op,
                self.inputs,
                input_to_check,
                output_names,
                delta=numeric_grad_delta,
                in_place=in_place) for input_to_check in inputs_to_check
        ]
        analytic_grads = self._get_gradient(inputs_to_check, place,
                                            output_names, no_grad_set)

        self.__assert_is_close(numeric_grads, analytic_grads, inputs_to_check,
                               max_relative_error,
                               "Gradient Check On %s" % str(place))

    @staticmethod
    def _create_var_descs_(block, var_dict):
        # FIXME: Try unify with `append_input_output`
        for param_name in var_dict:
            var = var_dict[param_name]
            if not isinstance(var, list) and not isinstance(var, tuple):
                var = [(param_name, var, None)]
            if not isinstance(var[0], list) and not isinstance(var[0], tuple):
                var = [(param_name, var[0], var[1])]

            for i, item in enumerate(var):
                if not isinstance(item[0], basestring):
                    item = [[param_name] + list(item)]
                if len(item) == 2:
                    if isinstance(item[1], tuple):
                        var[i] = [item[0], item[1][0], item[1][1]]
                    else:
                        # only set var name and value, set lod to None
                        var[i] = list(item) + [None]
            var_descs = [(block.create_var(
                name=name, shape=each.shape, dtype=each.dtype), each, lod)
                         for name, each, lod in var]

            yield param_name, var_descs

    @staticmethod
    def _merge_list(iterable):
        return reduce(lambda a, b: list(a) + list(b), iterable, [])

    @staticmethod
    def _numpy_to_lod_tensor(np_value, lod, place):
        tensor = core.LoDTensor()
        tensor.set(np_value, place)
        if lod is not None:
            tensor.set_lod(lod)
        return tensor

    @staticmethod
    def np_dtype_to_fluid_dtype(input):
        """Change the dtype of float16 numpy array

        numpy float16 is binded to paddle::platform::float16
        in tensor_py.h via the help of uint16 data type since
        the internal memory representation of float16 is
        uint16_t in paddle and np.uint16 in numpy, which are
        themselves binded together by pybind.

        Args:
            input: input numpy array

        Returns:
            input: The dtype of input will be changed to np.uint16 if
                it is originally np.float16, such that the internal memory
                of input will be reinterpreted as of dtype np.uint16.
        """
        if input.dtype == np.float16:
            input.dtype = np.uint16
        return input

    def _get_gradient(self, input_to_check, place, output_names, no_grad_set):
        prog = Program()
        block = prog.global_block()
        inputs_with_np = {
            key: value
            for (key, value) in OpTest._create_var_descs_(
                block, getattr(self, 'inputs', {}))
        }
        outputs_with_np = {
            key: val
            for (key, val) in OpTest._create_var_descs_(
                block, getattr(self, 'outputs', {}))
        }
        inputs = {
            k: [item[0] for item in inputs_with_np[k]]
            for k in inputs_with_np
        }
        outputs = {
            k: [item[0] for item in outputs_with_np[k]]
            for k in outputs_with_np
        }

        op = block.append_op(
            type=self.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=getattr(self, 'attrs', {}))

        # infer variable type and infer shape in compile-time
        op.desc.infer_var_type(block.desc)
        op.desc.infer_shape(block.desc)

        mean_inputs = map(block.var, output_names)

        if len(mean_inputs) == 1:
            loss = block.create_var(dtype=mean_inputs[0].dtype, shape=[1])
            op = block.append_op(
                inputs={"X": mean_inputs}, outputs={"Out": loss}, type='mean')
            op.desc.infer_var_type(block.desc)
            op.desc.infer_shape(block.desc)
        else:
            avg_sum = []
            for cur_loss in mean_inputs:
                cur_avg_loss = block.create_var(dtype=cur_loss.dtype, shape=[1])
                op = block.append_op(
                    inputs={"X": [cur_loss]},
                    outputs={"Out": [cur_avg_loss]},
                    type="mean")
                op.desc.infer_var_type(block.desc)
                op.desc.infer_shape(block.desc)
                avg_sum.append(cur_avg_loss)

            loss_sum = block.create_var(dtype=avg_sum[0].dtype, shape=[1])
            op_sum = block.append_op(
                inputs={"X": avg_sum}, outputs={"Out": loss_sum}, type='sum')
            op_sum.desc.infer_var_type(block.desc)
            op_sum.desc.infer_shape(block.desc)

            loss = block.create_var(dtype=loss_sum.dtype, shape=[1])
            op_loss = block.append_op(
                inputs={"X": loss_sum},
                outputs={"Out": loss},
                type='scale',
                attrs={'scale': 1.0 / float(len(avg_sum))})
            op_loss.desc.infer_var_type(block.desc)
            op_loss.desc.infer_shape(block.desc)

        param_grad_list = append_backward(
            loss=loss, parameter_list=input_to_check, no_grad_set=no_grad_set)

        feed_dict = {
            item[0].name: OpTest._numpy_to_lod_tensor(item[1], item[2], place)
            for p_name in inputs_with_np for item in inputs_with_np[p_name]
        }

        fetch_list = [g for p, g in param_grad_list]
        executor = Executor(place)
        return map(np.array,
                   executor.run(prog, feed_dict, fetch_list,
                                return_numpy=False))
