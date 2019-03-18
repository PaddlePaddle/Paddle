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

from __future__ import print_function

import numpy as np

import paddle.fluid.core as core
from paddle.fluid.op import Operator


def create_op(scope, op_type, inputs, outputs, attrs, cache_list=None):
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
    if cache_list != None and isinstance(cache_list, list):
        for name in cache_list:
            kwargs[name] = []
            scope.var(name)
            kwargs[name].append(name)

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
    def np_value_to_fluid_value(input):
        if input.dtype == np.float16:
            input = input.view(np.uint16)
        return input

    def __set_input__(var_name, var):
        if isinstance(var, tuple) or isinstance(var, np.ndarray):
            tensor = scope.find_var(var_name).get_tensor()
            if isinstance(var, tuple):
                tensor.set_recursive_sequence_lengths(var[1])
                var = var[0]
            tensor._set_dims(var.shape)
            tensor.set(np_value_to_fluid_value(var), place)
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


def append_input_output(block, op_proto, np_list, is_input, dtype):
    '''Insert VarDesc and generate Python variable instance'''
    proto_list = op_proto.inputs if is_input else op_proto.outputs

    def create_var(block, name, np_list, var_proto):
        dtype = None
        shape = None
        lod_level = None
        if name not in np_list:
            assert var_proto.intermediate, "{} not found".format(name)
        else:
            # inferece the dtype from numpy value.
            np_value = np_list[name]
            if isinstance(np_value, tuple):
                dtype = np_value[0].dtype
                # output shape, lod should be infered from input.
                if is_input:
                    shape = list(np_value[0].shape)
                    lod_level = len(np_value[1])
            else:
                dtype = np_value.dtype
                if is_input:
                    shape = list(np_value.shape)
                    lod_level = 0
        # NOTE(dzhwinter): type hacking
        # numpy float16 is binded to paddle::platform::float16
        # in tensor_py.h via the help of uint16 datatype. Because
        # the internal memory representation of float16 is
        # actually uint16_t in paddle. So we use np.uint16 in numpy for
        # raw memory, it can pass through the pybind. So in the testcase,
        # we feed data use data.view(uint16), but the dtype is float16 in fact.
        # The data.view(uint16) means do not cast the data type, but process data as the uint16
        if dtype == np.uint16:
            dtype = np.float16
        return block.create_var(
            dtype=dtype, shape=shape, lod_level=lod_level, name=name)

    var_dict = {}
    for var_proto in proto_list:
        var_name = str(var_proto.name)
        if (var_name not in np_list) and var_proto.dispensable:
            continue
        if is_input:
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


def append_loss_ops(block, output_names):
    mean_inputs = list(map(block.var, output_names))

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
    return loss
