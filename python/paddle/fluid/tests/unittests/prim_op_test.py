# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import struct
from collections import defaultdict

import config
import numpy as np

import paddle
import paddle.fluid.core as core
from paddle.fluid.framework import _dygraph_tracer, in_dygraph_mode
from paddle.fluid.layers.utils import map_structure
from paddle.jit.dy2static.utils import parse_arg_and_kwargs


def flatten(nest_list):
    out = []
    for i in nest_list:
        if isinstance(i, list or tuple):
            tmp_list = flatten(i)
            for j in tmp_list:
                out.append(j)
        else:
            out.append(i)
    return out


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, list or tuple) else [x]


def convert_uint16_to_float(in_list):
    in_list = np.asarray(in_list)
    out = np.vectorize(
        lambda x: struct.unpack(
            '<f', struct.pack('<I', np.uint32(x) << np.uint32(16))
        )[0],
        otypes=[np.float32],
    )(in_list.flat)
    return np.reshape(out, in_list.shape)


# TODO(wanghao107): OpTestUtils will be moved to op_test.py
class OpTestUtils:
    @classmethod
    def _get_kernel_signature(
        cls, op_type, eager_tensor_inputs, eager_tensor_outputs, attrs_outputs
    ):
        try:
            kernel_sig = _dygraph_tracer()._get_kernel_signature(
                op_type,
                eager_tensor_inputs,
                eager_tensor_outputs,
                attrs_outputs,
            )
        except RuntimeError as re:
            """we think the kernel_sig is missing."""
            kernel_sig = None
            print(
                "[Warning: op_test.py] Kernel Signature is not found for %s, fall back to intermediate state."
                % op_type
            )
        return kernel_sig

    @classmethod
    def prepare_python_api_arguments(
        cls, api, op_proto_ins, op_proto_attrs, kernel_sig
    ):
        """map from `op proto inputs and attrs` to `api input list and api attrs dict`

        NOTE: the op_proto_attrs and op_proto_ins is a default dict. default value is []
        """

        class Empty:
            pass

        def is_empty(a):
            return isinstance(a, Empty)

        def get_default(idx, defaults):
            assert not isinstance(defaults[idx], Empty), (
                "%d-th params of python api don't have default value." % idx
            )
            return defaults[idx]

        def to_defaults_list(params, defaults):
            return [defaults[p] for p in params if p in defaults]

        def parse_attri_value(name, op_inputs, op_attrs):
            """parse true value from inputs and attrs, if there is no name passed by OpTest, return Empty
            1. if the name in op_attrs, use the op_attrs[name]
            2. if the name in op_inputs, convert the op_inputs to [type of default value]
            3. if the name not in op_attrs ans op_inputs, return Empty. (this will use the default value from python api)
            """
            if name in op_proto_attrs:
                return op_proto_attrs[name]
            elif name in op_inputs:
                if len(op_inputs[name]) == 1:
                    # why don't use numpy().item() : if the Tensor is float64, we will change it to python.float32, where we loss accuracy: [allclose_op]
                    # why we reconstruct a tensor: because we want the tensor in cpu.
                    if in_dygraph_mode():
                        return paddle.to_tensor(
                            op_inputs[name][0].numpy(), place='cpu'
                        )
                    else:
                        return op_inputs[name][0]
                else:
                    # if this is a list (test_unsqueeze2_op): we just pass it into the python api.
                    return op_inputs[name]
            else:
                return Empty()

        # NOTE(xiongkun): the logic of constructing parameters:
        # for example:
        #    python api: cumprod(x, dim, dtype=None, name=None)
        #    kernel sig: [["x"], ["dim"], ["out"]]"
        #
        # we will construct a lot of list with the same length : len == len(api_params), here is 4
        #    api_params = ["x", "dim", "dtype", "name"]
        #    api_defaults = [Empty, Empty, None, None]; empty means no defaults.
        #    inputs_and_attrs = ["x", "dim"] , the length may shorter or longer than api_params
        #    input_arguments = [RealValue in self.inputs and self.attrs]
        # then ,we will loop for the api_params, construct a result list:
        #    if the name in ['name', 'dtype', 'out', 'output'], we will use the default value
        #    else, we will consume a input_arguments. (because the name is not corresponding, so we only use the order)

        api_params, api_defaults = parse_arg_and_kwargs(api)
        api_defaults = to_defaults_list(api_params, api_defaults)
        api_defaults = [
            Empty() for i in range(len(api_params) - len(api_defaults))
        ] + api_defaults
        assert len(api_defaults) == len(
            api_params
        ), "Error happens. contack xiongkun03 to solve."
        inputs_sig, attrs_sig, outputs_sig = kernel_sig
        inputs_and_attrs = inputs_sig + attrs_sig
        input_arguments = [
            op_proto_ins.get(name, Empty()) for name in inputs_sig
        ] + [
            parse_attri_value(name, op_proto_ins, op_proto_attrs)
            for name in attrs_sig
        ]
        results = []
        api_ignore_param_list = set(['name', 'dtype', 'out', 'output'])
        idx_of_op_proto_arguments = 0
        for idx, arg_name in enumerate(api_params):
            if arg_name in api_ignore_param_list:
                results.append(get_default(idx, api_defaults))
            else:
                if idx_of_op_proto_arguments < len(input_arguments):
                    tmp = input_arguments[idx_of_op_proto_arguments]
                    idx_of_op_proto_arguments += 1
                else:
                    tmp = Empty()  # use the default value

                if isinstance(tmp, Empty):
                    results.append(get_default(idx, api_defaults))
                else:
                    results.append(tmp)
        assert len(results) == len(api_params)
        return results

    @classmethod
    def assumption_assert_and_transform(cls, args, inp_num):
        """
        transform inputs by the following rules:
            1. [Tensor] -> Tensor
            2. [Tensor, Tensor, ...] -> list of Tensors
            3. None -> None
            4. Others: raise Error

        only support "X" is list of Tensor, currently don't support other structure like dict.
        """
        inp_args = [
            [inp] if inp is None else inp for inp in args[:inp_num]
        ]  # convert None -> [None]
        for inp in inp_args:
            assert isinstance(
                inp, list
            ), "currently only support `X` is [Tensor], don't support other structure."
        args = [inp[0] if len(inp) == 1 else inp for inp in inp_args] + args[
            inp_num:
        ]
        return args

    @classmethod
    def is_bfloat16_type(cls, np_type):
        if np_type == np.dtype('uint16'):
            return True
        return False


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


class PrimNet(paddle.nn.Layer):
    def __init__(self, python_api):
        super(PrimNet, self).__init__()
        self.python_api = python_api

    def forward(self, args):
        out = self.python_api(*args)
        return out


class PrimForwardChecker:
    def __init__(self, op_test, place):
        self.checker_name = "PrimForwardChecker"
        self.place = place
        self.op_test = op_test
        self.save_eager_or_static_status()
        self.init()
        self.init_checker()

    def init(self):
        pass

    def save_eager_or_static_status(self):
        self.eager_mode = True if in_dygraph_mode() else False

    def recover_eager_or_static_status(self):
        if self.eager_mode:
            paddle.disable_static()
        else:
            paddle.enable_static()

    def init_checker(self):
        assert hasattr(
            self.op_test, 'prim_op_type'
        ), "if you want to test comp op, please set prim_op_type with \'prim\' or \'comp\' in setUp function."
        assert self.op_test.prim_op_type in [
            "comp",
            "prim",
        ], "prim_op_type must be comp or prim in setUp function."
        assert hasattr(
            self.op_test, 'dtype'
        ), "Please set dtype in setUp function."
        self.op_type = self.op_test.op_type
        self.prim_op_type = self.op_test.prim_op_type
        self.python_api = self.op_test.python_api
        self.dtype = np.dtype(self.op_test.dtype)
        self.inputs = self.op_test.inputs
        self.attrs = (
            self.op_test.attrs if hasattr(self.op_test, 'attrs') else {}
        )
        self.outputs = self.op_test.outputs
        self.init_checker_threshold()
        self.enable_fw_comp = (
            self.op_test.enable_fw_comp
            if hasattr(self.op_test, 'enable_fw_comp')
            else True
        )
        self.enable_rev_comp = (
            self.op_test.enable_rev_comp
            if hasattr(self.op_test, 'enable_rev_comp')
            else True
        )
        self.enable_cinn = (
            self.op_test.enable_cinn
            if hasattr(self.op_test, 'enable_cinn')
            else True
        )
        self.enable_check_eager_comp = (
            self.op_test.enable_check_eager_comp
            if hasattr(self.op_test, 'enable_check_eager_comp')
            else True
        )
        self.enable_check_static_comp = (
            self.op_test.enable_check_static_comp
            if hasattr(self.op_test, 'enable_check_static_comp')
            else True
        )
        self.enable_check_jit_comp = (
            self.op_test.enable_check_jit_comp
            if hasattr(self.op_test, 'enable_check_jit_comp')
            else True
        )
        self.enable_check_jit_comp_with_cinn = (
            self.op_test.enable_check_jit_comp_with_cinn
            if hasattr(self.op_test, 'enable_check_jit_comp_with_cinn')
            else True
        )
        self.only_prim = (
            self.op_test.only_prim
            if hasattr(self.op_test, 'only_prim')
            else False
        )
        self.kernel_sig = self.get_kernel_sig()

    def init_checker_threshold(self):
        if hasattr(self.op_test, 'jit_comp_rtol'):
            self.jit_comp_rtol = self.op_test.jit_comp_rtol
        else:
            self.jit_comp_rtol = (
                config.TOLERANCE[self.dtype]['jit_comp']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'jit_comp_atol'):
            self.jit_comp_atol = self.op_test.jit_comp_atol
        else:
            self.jit_comp_atol = (
                config.TOLERANCE[self.dtype]['jit_comp']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'fw_comp_rtol'):
            self.fw_comp_rtol = self.op_test.fw_comp_rtol
        else:
            self.fw_comp_rtol = (
                config.TOLERANCE[self.dtype]['fw_comp']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'fw_comp_atol'):
            self.fw_comp_atol = self.op_test.fw_comp_atol
        else:
            self.fw_comp_atol = (
                config.TOLERANCE[self.dtype]['fw_comp']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'rev_comp_rtol'):
            self.rev_comp_rtol = self.op_test.rev_comp_rtol
        else:
            self.rev_comp_rtol = (
                config.TOLERANCE[self.dtype]['rev_comp']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'rev_comp_atol'):
            self.rev_comp_atol = self.op_test.rev_comp_atol
        else:
            self.rev_comp_atol = (
                config.TOLERANCE[self.dtype]['rev_comp']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'cinn_rtol'):
            self.cinn_rtol = self.op_test.cinn_rtol
        else:
            self.cinn_rtol = (
                config.TOLERANCE[self.dtype]['cinn']['rtol']
                if self.dtype in config.TOLERANCE
                else 0
            )

        if hasattr(self.op_test, 'cinn_atol'):
            self.cinn_atol = self.op_test.cinn_atol
        else:
            self.cinn_atol = (
                config.TOLERANCE[self.dtype]['cinn']['atol']
                if self.dtype in config.TOLERANCE
                else 0
            )

    def check(self):
        self.eager_desire = self.get_eager_desire()
        if self.enable_check_static_comp:
            self.check_static_comp()
        if self.enable_check_jit_comp:
            self.check_jit_comp()
        if self.enable_check_jit_comp_with_cinn:
            self.check_jit_comp_with_cinn()

        self.recover_eager_or_static_status()

    def get_kernel_sig(self):
        paddle.disable_static()
        if type(self.place) is paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) is paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        (
            eager_tensor_inputs,
            attrs_outputs,
            _,
        ) = self.get_eager_input_attr_and_inputdict()
        eager_tensor_outputs = self.get_eager_empty_output()
        kernel_sig = OpTestUtils._get_kernel_signature(
            self.op_type,
            eager_tensor_inputs,
            eager_tensor_outputs,
            attrs_outputs,
        )
        return kernel_sig

    def is_only_check_prim(self):
        return self.only_prim

    def get_eager_desire(self):
        paddle.disable_static()
        if type(self.place) is paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) is paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        (
            eager_tensor_inputs,
            attrs_outputs,
            _,
        ) = self.get_eager_input_attr_and_inputdict()
        args = OpTestUtils.prepare_python_api_arguments(
            self.python_api, eager_tensor_inputs, attrs_outputs, self.kernel_sig
        )
        inputs_sig, _, _ = self.kernel_sig
        args = OpTestUtils.assumption_assert_and_transform(
            args, len(inputs_sig)
        )
        ret = flatten(_as_list(self.python_api(*args)))
        ret = map_structure(lambda x: x.numpy(), ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            ret = map_structure(lambda x: convert_uint16_to_float(x), ret)
        return ret

    def get_eager_input_attr_and_inputdict(self):
        attrs_outputs = {}
        for attrs_name in self.attrs:
            if self.attrs[attrs_name] is not None:
                attrs_outputs[attrs_name] = self.attrs[attrs_name]
        input_dict = {}
        eager_inputs = defaultdict(list)
        for name, item in self.inputs.items():
            if isinstance(item, list):
                for tup in item:
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                        else tup[1].dtype
                    )
                    x = paddle.to_tensor(
                        data=tup[1],
                        place=self.place,
                        stop_gradient=False,
                        dtype=dtype,
                    )
                    eager_inputs[name].append(x)
                    input_dict.update({str(tup[0]): x})
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.to_tensor(
                    data=item,
                    place=self.place,
                    stop_gradient=False,
                    dtype=dtype,
                )
                eager_inputs[name].append(x)
                input_dict.update({name: x})
        return eager_inputs, attrs_outputs, input_dict

    def get_eager_empty_output(self):
        eager_outputs = defaultdict(list)
        for name, item in self.outputs.items():
            if isinstance(item, list):
                for tup in item:
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                        else tup[1].dtype
                    )
                    x = paddle.to_tensor(
                        data=[],
                        place=self.place,
                        stop_gradient=False,
                        dtype=dtype,
                    )
                    eager_outputs[name].append(x)
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.to_tensor(
                    data=[], place=self.place, stop_gradient=False, dtype=dtype
                )
                eager_outputs[name].append(x)
        return eager_outputs

    def get_static_input_attr_inputdict_and_feed(self):
        attrs_outputs = {}
        for attrs_name in self.attrs:
            if self.attrs[attrs_name] is not None:
                attrs_outputs[attrs_name] = self.attrs[attrs_name]
        input_dict = {}
        static_inputs = defaultdict(list)
        feed = {}
        for name, item in self.inputs.items():
            if isinstance(item, list):
                for tup in item:
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                        else tup[1].dtype
                    )
                    x = paddle.static.data(
                        name=str(tup[0]), shape=tup[1].shape, dtype=dtype
                    )
                    x.stop_gradient = False
                    static_inputs[name].append(x)
                    feed.update({str(tup[0]): tup[1]})
                    input_dict.update({str(tup[0]): x})
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.static.data(name=name, shape=item.shape, dtype=dtype)
                x.stop_gradient = False
                static_inputs[name].append(x)
                feed.update({name: item})
                input_dict.update({name: x})
        return static_inputs, attrs_outputs, input_dict, feed

    def check_eager_comp(self):
        pass

    def check_static_comp(self):
        # forward comp only for comp op
        if self.prim_op_type == "prim":
            return
        paddle.enable_static()
        core._set_prim_forward_enabled(self.enable_fw_comp)
        startup_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, startup_program):
            (
                static_inputs,
                attrs,
                input_dict,
                feed,
            ) = self.get_static_input_attr_inputdict_and_feed()
            args = OpTestUtils.prepare_python_api_arguments(
                self.python_api, static_inputs, attrs, self.kernel_sig
            )
            inputs_sig, _, _ = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            ret = flatten(_as_list(self.python_api(*args)))
            paddle.incubate.autograd.to_prim(main_program.blocks)
        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)
        ret = exe.run(main_program, feed=feed, fetch_list=ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            ret = map_structure(lambda x: convert_uint16_to_float(x), ret)
        # check static forward
        if len(ret) != len(self.eager_desire):
            msg = (
                "The static comp forward api out tensor nums is different with eager forward api out tensor nums on %s."
                'when enable_fw_comp is %s, static comp forward api out tensor nums = %s, eager forward api out tensor nums = %s. \n'
                % (
                    str(self.place),
                    self.enable_fw_comp,
                    len(ret),
                    len(self.eager_desire),
                )
            )
            raise RuntimeError(msg)
        for i in range(len(ret)):
            if not np.allclose(
                ret[i],
                self.eager_desire[i],
                rtol=self.fw_comp_rtol,
                atol=self.fw_comp_atol,
            ):
                msg = (
                    'Check static comp forward api out failed. Mismatch between static comp '
                    'and eager on %s, when enable_fw_comp is %s,the forward api out tensor\'s index is : %d \n'
                    'static comp forward api out tensor:%s\n eager forward api out tensor:%s\n'
                    % (
                        str(self.place),
                        self.enable_fw_comp,
                        i,
                        ret[i],
                        self.eager_desire[i],
                    )
                )
                raise RuntimeError(msg)
        paddle.disable_static()
        core._set_prim_forward_enabled(False)

    def check_jit_comp(self):
        if self.prim_op_type == "prim":
            return
        paddle.disable_static()
        if type(self.place) == paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) == paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        atol = self.fw_comp_atol if self.enable_fw_comp else self.jit_comp_atol
        rtol = self.fw_comp_rtol if self.enable_fw_comp else self.jit_comp_rtol
        core._set_prim_forward_enabled(self.enable_fw_comp)
        (
            eager_tensor_inputs,
            attrs_outputs,
            _,
        ) = self.get_eager_input_attr_and_inputdict()
        args = OpTestUtils.prepare_python_api_arguments(
            self.python_api, eager_tensor_inputs, attrs_outputs, self.kernel_sig
        )
        inputs_sig, _, _ = self.kernel_sig
        args = OpTestUtils.assumption_assert_and_transform(
            args, len(inputs_sig)
        )
        net = PrimNet(self.python_api)
        net = apply_to_static(net, False)
        ret = flatten(_as_list(net(args)))
        ret = map_structure(lambda x: x.numpy(), ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            ret = map_structure(lambda x: convert_uint16_to_float(x), ret)
        # check jit comp forward
        if len(ret) != len(self.eager_desire):
            msg = (
                "The jit comp forward api out tensor nums is different with eager forward api out tensor nums on %s."
                'when enable_fw_comp is %s, jit comp forward api out tensor nums = %s, eager forward api out tensor nums = %s. \n'
                % (
                    str(self.place),
                    self.enable_fw_comp,
                    len(ret),
                    len(self.eager_desire),
                )
            )
            raise RuntimeError(msg)
        for i in range(len(ret)):
            if not np.allclose(
                ret[i], self.eager_desire[i], rtol=rtol, atol=atol
            ):
                msg = (
                    'Check jit comp forward api out failed. Mismatch between jit comp '
                    'and eager on %s, when enable_fw_comp is %s,the forward api out tensor\'s index is : %d \n'
                    'jit comp forward api out tensor:%s\n eager forward api out tensor:%s\n'
                    % (
                        str(self.place),
                        self.enable_fw_comp,
                        i,
                        ret[i],
                        self.eager_desire[i],
                    )
                )
                raise RuntimeError(msg)
        core._set_prim_forward_enabled(False)
        net.forward.program_cache.clear()

    def check_jit_comp_with_cinn(self):
        if self.prim_op_type == "prim":
            return
        # cinn doesn't suppoort cpu place
        if (
            type(self.place) == paddle.fluid.libpaddle.CPUPlace
            and self.enable_cinn
            and core.is_compiled_with_cinn()
        ):
            return
        paddle.disable_static()
        atol = (
            self.cinn_atol
            if self.enable_cinn and core.is_compiled_with_cinn()
            else self.fw_comp_atol
        )
        rtol = (
            self.cinn_rtol
            if self.enable_cinn and core.is_compiled_with_cinn()
            else self.fw_comp_rtol
        )
        core._set_prim_forward_enabled(self.enable_fw_comp)
        if type(self.place) is paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) is paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        (
            eager_tensor_inputs,
            attrs_outputs,
            _,
        ) = self.get_eager_input_attr_and_inputdict()
        args = OpTestUtils.prepare_python_api_arguments(
            self.python_api, eager_tensor_inputs, attrs_outputs, self.kernel_sig
        )
        inputs_sig, _, _ = self.kernel_sig
        args = OpTestUtils.assumption_assert_and_transform(
            args, len(inputs_sig)
        )
        net = PrimNet(self.python_api)
        net = apply_to_static(
            net, core.is_compiled_with_cinn() and self.enable_cinn
        )
        ret = flatten(_as_list(net(args)))
        ret = map_structure(lambda x: x.numpy(), ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            ret = map_structure(lambda x: convert_uint16_to_float(x), ret)
        # check jit comp forward
        if len(ret) != len(self.eager_desire):
            msg = (
                "The jit comp with cinn forward api out tensor nums is different with eager forward api out tensor nums on %s."
                'when enable_fw_comp is %s, enable_cinn is %s, jit comp forward api out tensor nums = %s, eager forward api out tensor nums = %s. \n'
                % (
                    str(self.place),
                    self.enable_fw_comp,
                    core.is_compiled_with_cinn() and self.enable_cinn,
                    len(ret),
                    len(self.eager_desire),
                )
            )
            raise RuntimeError(msg)
        for i in range(len(ret)):
            if not np.allclose(
                ret[i], self.eager_desire[i], rtol=rtol, atol=atol
            ):
                msg = (
                    'Check jit comp with cinn forward api out failed. Mismatch between jit comp and eager on %s, '
                    'when enable_fw_comp is %s, enable_cinn is %s, the forward api out tensor\'s index is : %d \n'
                    'jit comp forward api out tensor:%s\n eager forward api out tensor:%s\n'
                    % (
                        str(self.place),
                        self.enable_fw_comp,
                        core.is_compiled_with_cinn() and self.enable_cinn,
                        i,
                        ret[i],
                        self.eager_desire[i],
                    )
                )
                raise RuntimeError(msg)
        core._set_prim_forward_enabled(False)
        net.forward.program_cache.clear()


class PrimGradChecker(PrimForwardChecker):
    def __init__(
        self,
        op_test,
        place,
        inputs_to_check,
        output_names,
        no_grad_set,
        grad_outputs,
    ):
        PrimForwardChecker.__init__(self, op_test, place)
        self.inputs_to_check = inputs_to_check
        self.output_names = output_names
        self.no_grad_set = no_grad_set
        self.grad_outputs = grad_outputs

    def init(self):
        self.checker_name = "PrimGradChecker"

    def check(self):
        self.eager_desire = self.get_eager_desire()
        if self.enable_check_eager_comp:
            self.check_eager_comp()
        if self.enable_check_static_comp:
            self.check_static_comp()
        if self.enable_check_jit_comp:
            self.check_jit_comp()
        if self.enable_check_jit_comp_with_cinn:
            self.check_jit_comp_with_cinn()

        self.recover_eager_or_static_status()

    def get_output_dict(self, np_outputs, api_outputs, outputs_sig):
        assert len(api_outputs) <= len(outputs_sig), (
            "forward api outputs length must be the less than or equal to KernelSignature outputs,but recive %s and %s"
        ) % (len(api_outputs), len(outputs_sig))
        output_dict = {}
        for i in range(len(api_outputs)):
            output_name = outputs_sig[i]
            if isinstance(np_outputs[output_name], list):
                for j, tup in enumerate(np_outputs[output_name]):
                    output_dict.update({tup[0]: api_outputs[i][j]})
            else:
                output_dict.update({output_name: api_outputs[i]})
        return output_dict

    def gen_eager_grad_outputs(self):
        if self.grad_outputs is None:
            return None
        eager_vs = []
        for np_v in self.grad_outputs:
            eager_vs.append(
                paddle.to_tensor(
                    data=np_v,
                    place=self.place,
                    dtype="bfloat16"
                    if OpTestUtils.is_bfloat16_type(np_v.dtype)
                    else np_v.dtype,
                )
            )
        return eager_vs

    def gen_static_grad_outputs_and_feed(self):
        if self.grad_outputs is None:
            return None, {}
        static_vs = []
        feed = {}
        for i, np_v in enumerate(self.grad_outputs):
            static_vs.append(
                paddle.static.data(
                    name='v_' + str(i),
                    shape=np_v.shape,
                    dtype="bfloat16"
                    if OpTestUtils.is_bfloat16_type(np_v.dtype)
                    else np_v.dtype,
                )
            )
            feed.update({'v_' + str(i): np_v})
        return static_vs, feed

    def gen_no_grad_set(self, var_dict):
        if self.no_grad_set is None:
            return None
        no_grad_set = set()
        for name in self.no_grad_set:
            if name in var_dict:
                no_grad_set.add(var_dict[name])
        return no_grad_set

    def get_eager_desire(self):
        paddle.disable_static()
        if type(self.place) is paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) is paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        (
            eager_tensor_inputs,
            attrs_outputs,
            inputs_dict,
        ) = self.get_eager_input_attr_and_inputdict()
        args = OpTestUtils.prepare_python_api_arguments(
            self.python_api, eager_tensor_inputs, attrs_outputs, self.kernel_sig
        )
        inputs_sig, _, outputs_sig = self.kernel_sig
        args = OpTestUtils.assumption_assert_and_transform(
            args, len(inputs_sig)
        )
        ret = _as_list(self.python_api(*args))
        outputs_dict = self.get_output_dict(self.outputs, ret, outputs_sig)
        ys = []
        if isinstance(self.output_names, list):
            for output_name in self.output_names:
                ys.append(outputs_dict[output_name])
        else:
            ys.append(outputs_dict[self.output_names])
        xs = []
        if isinstance(self.inputs_to_check, list):
            for input_name in self.inputs_to_check:
                xs.append(inputs_dict[input_name])
        else:
            xs.append(inputs_dict[self.inputs_to_check])
        vs = self.gen_eager_grad_outputs()
        no_grad_vars = self.gen_no_grad_set(
            var_dict={**inputs_dict, **outputs_dict}
        )
        ret = paddle.grad(
            ys, xs, vs, allow_unused=True, no_grad_vars=no_grad_vars
        )
        ret = map_structure(lambda x: x.numpy(), ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            ret = map_structure(lambda x: convert_uint16_to_float(x), ret)
        return ret

    def check_eager_comp(self):
        if self.prim_op_type == "comp":
            return
        paddle.disable_static()
        if type(self.place) is paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) is paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        atol = self.rev_comp_atol
        rtol = self.rev_comp_rtol
        core._set_prim_backward_enabled(self.enable_rev_comp)
        actual_ret = self.get_eager_desire()
        # check static forward
        if len(actual_ret) != len(self.eager_desire):
            msg = (
                "The eager comp grad out tensor nums is different with eager grad out tensor nums on %s."
                'when enable_rev_comp is %s, eager comp grad api out tensor nums = %s, eager grad out tensor nums = %s. \n'
                % (
                    str(self.place),
                    self.enable_rev_comp,
                    len(actual_ret),
                    len(self.eager_desire),
                )
            )
            raise RuntimeError(msg)
        for i in range(len(actual_ret)):
            if not np.allclose(
                actual_ret[i],
                self.eager_desire[i],
                rtol=atol,
                atol=rtol,
            ):
                msg = (
                    'Check eager comp grad out failed. Mismatch between eager comp '
                    'and eager on %s, when enable_rev_comp is %s,the eager comp grad out tensor\'s index is : %d \n'
                    'eager comp grad out tensor:%s\n eager grad out tensor:%s\n'
                    % (
                        str(self.place),
                        self.enable_rev_comp,
                        i,
                        actual_ret[i],
                        self.eager_desire[i],
                    )
                )
                raise RuntimeError(msg)

    def check_static_comp(self):
        paddle.enable_static()
        if self.prim_op_type == "prim":
            core._set_prim_backward_enabled(self.enable_rev_comp)
        else:
            core._set_prim_forward_enabled(self.enable_fw_comp)
            core._set_prim_backward_enabled(self.enable_rev_comp)
        atol = self.rev_comp_atol if self.enable_rev_comp else self.fw_comp_atol
        rtol = self.rev_comp_rtol if self.enable_rev_comp else self.fw_comp_rtol
        startup_program, main_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, startup_program):
            (
                static_inputs,
                attrs,
                inputs_dict,
                feed,
            ) = self.get_static_input_attr_inputdict_and_feed()
            args = OpTestUtils.prepare_python_api_arguments(
                self.python_api, static_inputs, attrs, self.kernel_sig
            )
            inputs_sig, _, outputs_sig = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            fw_outs = _as_list(self.python_api(*args))
            outputs_dict = self.get_output_dict(
                self.outputs, fw_outs, outputs_sig
            )
            paddle.incubate.autograd.to_prim(main_program.blocks)
            ys = []
            if isinstance(self.output_names, list):
                for output_name in self.output_names:
                    ys.append(outputs_dict[output_name])
            else:
                ys.append(outputs_dict[self.output_names])
            xs = []
            if isinstance(self.inputs_to_check, list):
                for input_name in self.inputs_to_check:
                    xs.append(inputs_dict[input_name])
            else:
                xs.append(inputs_dict[self.inputs_to_check])
            vs, vs_feed = self.gen_static_grad_outputs_and_feed()
            feed.update(vs_feed)
            no_grad_vars = self.gen_no_grad_set(
                var_dict={**inputs_dict, **outputs_dict}
            )
            ret = paddle.static.gradients(ys, xs, vs, no_grad_set=no_grad_vars)
        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)
        actual_ret = exe.run(main_program, feed=feed, fetch_list=ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            actual_ret = map_structure(
                lambda x: convert_uint16_to_float(x), actual_ret
            )
        # check static grad out
        if len(actual_ret) != len(self.eager_desire):
            msg = (
                "The static comp grad out tensor nums is different with eager grad out tensor nums on %s."
                'when enable_fw_comp is %s,enable_rev_comp is %s, static comp grad out tensor nums = %s, eager grad out tensor nums = %s. \n'
                % (
                    str(self.place),
                    self.enable_fw_comp,
                    self.enable_rev_comp,
                    len(actual_ret),
                    len(self.eager_desire),
                )
            )
            raise RuntimeError(msg)
        for i in range(len(actual_ret)):
            if not np.allclose(
                actual_ret[i], self.eager_desire[i], rtol=rtol, atol=atol
            ):
                msg = (
                    'Check static comp grad out failed. Mismatch between static comp '
                    'and eager on %s, when enable_fw_comp is %s,enable_rev_comp is %s,the forward api out tensor\'s index is : %d \n'
                    'static comp grad out tensor:%s\n eager grad out tensor:%s\n'
                    % (
                        str(self.place),
                        self.enable_fw_comp,
                        self.enable_rev_comp,
                        i,
                        actual_ret[i],
                        self.eager_desire[i],
                    )
                )
                raise RuntimeError(msg)
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)
        paddle.disable_static()

    def check_jit_comp(self):
        paddle.disable_static()
        if type(self.place) is paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) is paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        if self.prim_op_type == "prim":
            core._set_prim_backward_enabled(self.enable_rev_comp)
        else:
            core._set_prim_forward_enabled(self.enable_fw_comp)
            core._set_prim_backward_enabled(self.enable_rev_comp)
        atol = (
            self.fw_comp_atol
            if self.enable_fw_comp and not self.enable_rev_comp
            else self.jit_comp_atol
        )
        rtol = (
            self.fw_comp_rtol
            if self.enable_fw_comp and not self.enable_rev_comp
            else self.jit_comp_rtol
        )
        atol = self.rev_comp_atol if self.enable_rev_comp else atol
        rtol = self.rev_comp_rtol if self.enable_rev_comp else rtol
        (
            eager_tensor_inputs,
            attrs_outputs,
            inputs_dict,
        ) = self.get_eager_input_attr_and_inputdict()
        args = OpTestUtils.prepare_python_api_arguments(
            self.python_api, eager_tensor_inputs, attrs_outputs, self.kernel_sig
        )
        inputs_sig, _, outputs_sig = self.kernel_sig
        args = OpTestUtils.assumption_assert_and_transform(
            args, len(inputs_sig)
        )
        net = PrimNet(self.python_api)
        net = apply_to_static(net, False)
        out = _as_list(net(args))
        outputs_dict = self.get_output_dict(self.outputs, out, outputs_sig)
        ys = []
        if isinstance(self.output_names, list):
            for output_name in self.output_names:
                ys.append(outputs_dict[output_name])
        else:
            ys.append(outputs_dict[self.output_names])
        xs = []
        if isinstance(self.inputs_to_check, list):
            for input_name in self.inputs_to_check:
                xs.append(inputs_dict[input_name])
        else:
            xs.append(inputs_dict[self.inputs_to_check])
        vs = self.gen_eager_grad_outputs()
        no_grad_vars = self.gen_no_grad_set(
            var_dict={**inputs_dict, **outputs_dict}
        )
        ret = paddle.grad(
            ys, xs, vs, allow_unused=True, no_grad_vars=no_grad_vars
        )
        ret = map_structure(lambda x: x.numpy(), ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            ret = map_structure(lambda x: convert_uint16_to_float(x), ret)
        # check jit comp grad out
        if len(ret) != len(self.eager_desire):
            msg = (
                "The jit comp grad out tensor nums is different with eager grad out tensor nums on %s."
                'when enable_fw_comp is %s, enable_rev_comp is %s, jit comp grad out tensor nums = %s, eager grad out tensor nums = %s. \n'
                % (
                    str(self.place),
                    self.enable_fw_comp,
                    self.enable_rev_comp,
                    len(ret),
                    len(self.eager_desire),
                )
            )
            raise RuntimeError(msg)
        for i in range(len(ret)):
            if not np.allclose(
                ret[i], self.eager_desire[i], rtol=rtol, atol=atol
            ):
                msg = (
                    'Check jit comp grad out failed. Mismatch between jit comp '
                    'and eager on %s, when enable_fw_comp is %s, enable_rev_comp is %s,the grad out tensor\'s index is : %d \n'
                    'jit comp grad out tensor:%s\n eager grad out out tensor:%s\n'
                    % (
                        str(self.place),
                        self.enable_fw_comp,
                        self.enable_rev_comp,
                        i,
                        ret[i],
                        self.eager_desire[i],
                    )
                )
                raise RuntimeError(msg)
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)
        net.forward.program_cache.clear()

    def check_jit_comp_with_cinn(self):
        # cinn doesen't support cpu place
        if (
            type(self.place) is paddle.fluid.libpaddle.CPUPlace
            and self.enable_cinn
            and core.is_compiled_with_cinn()
        ):
            return
        paddle.disable_static()
        if type(self.place) is paddle.fluid.libpaddle.CPUPlace:
            paddle.device.set_device("cpu")
        if type(self.place) is paddle.fluid.libpaddle.CUDAPlace:
            paddle.device.set_device("gpu:0")
        if self.prim_op_type == "prim":
            core._set_prim_backward_enabled(self.enable_rev_comp)
        else:
            core._set_prim_forward_enabled(self.enable_fw_comp)
            core._set_prim_backward_enabled(self.enable_rev_comp)
        if self.enable_cinn and core.is_compiled_with_cinn():
            atol = self.cinn_atol
            rtol = self.cinn_rtol
        else:
            atol = (
                self.fw_comp_atol
                if self.enable_fw_comp and not self.enable_rev_comp
                else self.jit_comp_atol
            )
            rtol = (
                self.fw_comp_rtol
                if self.enable_fw_comp and not self.enable_rev_comp
                else self.jit_comp_rtol
            )
            atol = self.rev_comp_atol if self.enable_rev_comp else atol
            rtol = self.rev_comp_rtol if self.enable_rev_comp else rtol
        (
            eager_tensor_inputs,
            attrs_outputs,
            inputs_dict,
        ) = self.get_eager_input_attr_and_inputdict()
        args = OpTestUtils.prepare_python_api_arguments(
            self.python_api, eager_tensor_inputs, attrs_outputs, self.kernel_sig
        )
        inputs_sig, _, outputs_sig = self.kernel_sig
        args = OpTestUtils.assumption_assert_and_transform(
            args, len(inputs_sig)
        )
        net = PrimNet(self.python_api)
        net = apply_to_static(
            net, core.is_compiled_with_cinn() and self.enable_cinn
        )
        out = _as_list(net(args))
        outputs_dict = self.get_output_dict(self.outputs, out, outputs_sig)
        ys = []
        if isinstance(self.output_names, list):
            for output_name in self.output_names:
                ys.append(outputs_dict[output_name])
        else:
            ys.append(outputs_dict[self.output_names])
        xs = []
        if isinstance(self.inputs_to_check, list):
            for input_name in self.inputs_to_check:
                xs.append(inputs_dict[input_name])
        else:
            xs.append(inputs_dict[self.inputs_to_check])
        vs = self.gen_eager_grad_outputs()
        no_grad_vars = self.gen_no_grad_set(
            var_dict={**inputs_dict, **outputs_dict}
        )
        ret = paddle.grad(
            ys, xs, vs, allow_unused=True, no_grad_vars=no_grad_vars
        )
        ret = map_structure(lambda x: x.numpy(), ret)
        if OpTestUtils.is_bfloat16_type(self.dtype):
            ret = map_structure(lambda x: convert_uint16_to_float(x), ret)
        # check jit comp grad out
        if len(ret) != len(self.eager_desire):
            msg = (
                "The jit comp with cinn grad out tensor nums is different with eager grad out tensor nums on %s."
                'when enable_fw_comp is %s, enable_rev_comp is %s, enable_cinn is %s, jit comp grad out tensor nums = %s, eager grad out tensor nums = %s. \n'
                % (
                    str(self.place),
                    self.enable_fw_comp,
                    self.enable_rev_comp,
                    self.enable_cinn and core.is_compiled_with_cinn(),
                    len(ret),
                    len(self.eager_desire),
                )
            )
            raise RuntimeError(msg)
        for i in range(len(ret)):
            if not np.allclose(
                ret[i], self.eager_desire[i], rtol=rtol, atol=atol
            ):
                msg = (
                    'Check jit comp with cinn grad out failed. Mismatch between jit comp with cinn '
                    'and eager on %s, when enable_fw_comp is %s, enable_rev_comp is %s, enable_cinn is %s,'
                    'the grad out tensor\'s index is : %d ,jit comp with cinn grad out tensor:%s\n eager grad out out tensor:%s\n'
                    % (
                        str(self.place),
                        self.enable_fw_comp,
                        self.enable_rev_comp,
                        self.enable_cinn and core.is_compiled_with_cinn(),
                        i,
                        ret[i],
                        self.eager_desire[i],
                    )
                )
                raise RuntimeError(msg)
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)
        net.forward.program_cache.clear()
