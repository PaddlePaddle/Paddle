import unittest
import numpy as np
import random
import itertools
import paddle.v2.framework.core as core
import collections
from paddle.v2.framework.backward import append_backward_ops
from paddle.v2.framework.op import Operator
from paddle.v2.framework.executor import Executor
from paddle.v2.framework.framework import Program, OpProtoHolder


def create_op(scope, op_type, inputs, outputs, attrs):
    kwargs = dict()

    def __create_var__(name, var_name):
        scope.var(var_name).get_tensor()
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


class SendRecvTest(unittest.TestCase):
    def feed_var(self, input_vars, place):
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.LoDTensor()
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

    def check_output_with_place(self, place, atol):
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
        outs = exe.run(program, feed=feed_map, fetch_list=fetch_list)

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

    def check_output(self, atol=1e-5):
        places = [core.CPUPlace()]
        if core.is_compile_gpu() and core.op_support_gpu(self.op_type):
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
                        "the first error element is %d, %f, %f") % (
                            msg_prefix, name, max_diff, max_relative_error,
                            offset, a.flatten()[offset], b.flatten()[offset])

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

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
