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

from collections import defaultdict

import numpy as np
from prim_op_test import OpTestUtils, _as_list, convert_uint16_to_float, flatten
from utils import dygraph_guard

import paddle
import paddle.distributed as dist

TOLERANCE = {
    np.dtype('float64'): {"rtol": 1e-15, "atol": 0},
    np.dtype('float32'): {"rtol": 1e-6, "atol": 0},
    np.dtype('float16'): {"rtol": 1e-3, "atol": 0},
    np.dtype('uint16'): {"rtol": 1e-2, "atol": 0},
}


class AutoParallelForwardChecker:
    def __init__(
        self,
        op_type,
        pthon_api,
        dtype,
        input_specs,
        inputs,
        attrs,
        outputs,
        place,
        python_out_sig=None,
    ):
        self.checker_name = "AutoParallelForwardChecker"
        self.init_checker(
            op_type,
            pthon_api,
            dtype,
            input_specs,
            inputs,
            attrs,
            outputs,
            place,
            python_out_sig,
        )

    def init_checker(
        self,
        op_type,
        pthon_api,
        dtype,
        input_specs,
        inputs,
        attrs,
        outputs,
        place,
        python_out_sig=None,
    ):
        self.op_type = op_type
        self.public_python_api = pthon_api
        self.dtype = np.dtype(dtype)
        self.input_specs = input_specs
        self.inputs = inputs
        self.attrs = attrs
        self.outputs = outputs
        self.place = place
        if self.place == "cpu":
            paddle.device.set_device("cpu")
        if self.place == "gpu":
            paddle.device.set_device("gpu:" + str(dist.get_rank()))
        self.python_out_sig = python_out_sig
        self.attrs = attrs
        self.outputs = outputs
        self.init_checker_threshold()
        self.kernel_sig = self.get_kernel_sig()
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def init_checker_threshold(self, atol=None, rtol=None):
        self.atol = atol if atol else TOLERANCE[self.dtype]["atol"]
        self.rtol = rtol if rtol else TOLERANCE[self.dtype]["rtol"]

    def check(self):
        self.eager_forward_desire = self.get_eager_desire()
        self.check_eager_auto_parallel()

    def check_eager_auto_parallel(self):
        with dygraph_guard():
            actual_ret = self.get_eager_desire(dist_mode=True)
            # check eager auto parallel forward
            if len(actual_ret) != len(self.eager_forward_desire):
                msg = (
                    "The eager auto parallel out tensor nums is different with eager out tensor nums on {}."
                    'eager auto parallel out tensor nums = {}, eager out tensor nums = {}. \n'.format(
                        str(self.place),
                        len(actual_ret),
                        len(self.eager_forward_desire),
                    )
                )
                raise RuntimeError(msg)
            for i in range(len(actual_ret)):
                np.testing.assert_allclose(
                    actual_ret[i],
                    self.eager_forward_desire[i],
                    rtol=self.atol,
                    atol=self.rtol,
                    err_msg=(
                        'Check eager auto parallel failed. Mismatch between eager auto parallel outputs '
                        'and eager outputs on %s, the eager forward output tensor\'s index is : %d \n'
                        'eager auto parallel output tensor:\n%s\n eager output tensor:\n%s\n'
                        % (
                            str(self.place),
                            i,
                            actual_ret[i],
                            self.eager_forward_desire[i],
                        )
                    ),
                )

    def get_kernel_sig(self):
        with dygraph_guard():
            (
                eager_tensor_inputs,
                attrs_outputs,
                _,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=True)
            eager_tensor_outputs = self.get_eager_empty_output(
                stop_gradient=True
            )
            kernel_sig = OpTestUtils._get_kernel_signature(
                self.op_type,
                eager_tensor_inputs,
                eager_tensor_outputs,
                attrs_outputs,
            )
        return kernel_sig

    def get_eager_desire(self, dist_mode=False):
        with dygraph_guard():
            if dist_mode:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    _,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=True, dist_mode=True
                )
            else:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    _,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=True, dist_mode=False
                )
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
            )
            inputs_sig, _, _ = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            ret = flatten(_as_list(self.public_python_api(*args)))
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
        return ret

    def get_eager_input_attr_and_inputdict(
        self, stop_gradient, dist_mode=False
    ):
        attrs_outputs = {}
        for attrs_name in self.attrs:
            if self.attrs[attrs_name] is not None:
                attrs_outputs[attrs_name] = self.attrs[attrs_name]
        input_dict = {}
        eager_inputs = defaultdict(list)
        for name, item in self.inputs.items():
            if isinstance(item, list):
                if not dist_mode or name not in self.input_specs:
                    for tup in item:
                        dtype = (
                            "bfloat16"
                            if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                            else tup[1].dtype
                        )
                        x = paddle.to_tensor(
                            data=tup[1],
                            stop_gradient=stop_gradient,
                            dtype=dtype,
                        )
                        eager_inputs[name].append(x)
                        input_dict.update({str(tup[0]): x})
                else:
                    for i in range(len(item)):
                        dtype = (
                            "bfloat16"
                            if OpTestUtils.is_bfloat16_type(item[i][1].dtype)
                            else item[i][1].dtype
                        )
                        x_dist_attr = dist.DistAttr(
                            mesh=self._mesh,
                            sharding_specs=self.input_specs[name][i],
                        )
                        x = paddle.to_tensor(
                            data=item[i][1],
                            stop_gradient=stop_gradient,
                            dtype=dtype,
                        )
                        dist_x = dist.shard_tensor(x, x_dist_attr)
                        dist_x.stop_gradient = stop_gradient
                        eager_inputs[name].append(dist_x)
                        input_dict.update({str(item[i][0]): dist_x})
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.to_tensor(
                    data=item,
                    stop_gradient=stop_gradient,
                    dtype=dtype,
                )
                if not dist_mode or name not in self.input_specs:
                    eager_inputs[name].append(x)
                    input_dict.update({name: x})
                else:
                    x_dist_attr = dist.DistAttr(
                        mesh=self._mesh, sharding_specs=self.input_specs[name]
                    )
                    dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
                    dist_x.stop_gradient = stop_gradient
                    eager_inputs[name].append(dist_x)
                    input_dict.update({name: dist_x})
        return eager_inputs, attrs_outputs, input_dict

    def get_eager_empty_output(self, stop_gradient):
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
                        stop_gradient=stop_gradient,
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
                    data=[],
                    stop_gradient=stop_gradient,
                    dtype=dtype,
                )
                eager_outputs[name].append(x)
        return eager_outputs


class AutoParallelGradChecker(AutoParallelForwardChecker):
    def __init__(
        self,
        op_type,
        pthon_api,
        dtype,
        input_specs,
        inputs,
        attrs,
        outputs,
        place,
        inputs_to_check,
        output_names,
        no_grad_set,
        grad_outputs,
        python_out_sig=None,
    ):
        super().__init__(
            op_type,
            pthon_api,
            dtype,
            input_specs,
            inputs,
            attrs,
            outputs,
            place,
            python_out_sig,
        )
        self.checker_name = "AutoParallelGradChecker"
        self.inputs_to_check = inputs_to_check
        self.output_names = output_names
        self.no_grad_set = no_grad_set
        self.grad_outputs = grad_outputs

    def check(self):
        self.eager_grad_desire = self.get_eager_desire()
        self.check_eager_auto_parallel()

    def check_eager_auto_parallel(self):
        with dygraph_guard():
            actual_grad_ret = self.get_eager_desire(dist_mode=True)
            # check eager auto parallel forward
            if len(actual_grad_ret) != len(self.eager_grad_desire):
                msg = (
                    "The eager auto parallel grad out tensor nums is different with eager grad out tensor nums on {}."
                    'eager auto parallel grad out tensor nums = {}, eager grad out tensor nums = {}. \n'.format(
                        str(self.place),
                        len(actual_grad_ret),
                        len(self.eager_grad_desire),
                    )
                )
                raise RuntimeError(msg)
            for i in range(len(actual_grad_ret)):
                np.testing.assert_allclose(
                    actual_grad_ret[i],
                    self.eager_grad_desire[i],
                    rtol=self.atol,
                    atol=self.rtol,
                    err_msg=(
                        'Check eager auto parallel backward failed. Mismatch between eager auto parallel grad outputs '
                        'and eager grad outputs on %s, the eager grad output tensor\'s index is : %d \n'
                        'eager auto parallel grad output tensor:\n%s\n eager grad output tensor:\n%s\n'
                        % (
                            str(self.place),
                            i,
                            actual_grad_ret[i],
                            self.eager_grad_desire[i],
                        )
                    ),
                )

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

    def get_output_dict(self, np_outputs, api_outputs, outputs_sig):
        assert len(api_outputs) <= len(outputs_sig), (
            "forward api outputs length must be the less than or equal to KernelSignature outputs,but receive {} and {}"
        ).format(len(api_outputs), len(outputs_sig))
        output_dict = {}
        for i in range(len(api_outputs)):
            output_name = outputs_sig[i]
            if output_name in np_outputs and isinstance(
                np_outputs[output_name], list
            ):
                for j, tup in enumerate(np_outputs[output_name]):
                    output_dict.update({tup[0]: api_outputs[i][j]})
            else:
                output_dict.update({output_name: api_outputs[i]})
        return output_dict

    def gen_no_grad_set(self, var_dict):
        if self.no_grad_set is None:
            return None
        no_grad_set = set()
        for name in self.no_grad_set:
            if name in var_dict:
                no_grad_set.add(var_dict[name])
        return no_grad_set

    def get_eager_desire(self, dist_mode=False):
        with dygraph_guard():
            if dist_mode:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    inputs_dict,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=False, dist_mode=True
                )
            else:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    inputs_dict,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=False, dist_mode=False
                )
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
            )
            inputs_sig, _, outputs_sig = self.kernel_sig
            if self.python_out_sig is not None:
                outputs_sig = self.python_out_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )

            ret = _as_list(self.public_python_api(*args))
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
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
        return ret
