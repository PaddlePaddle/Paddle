#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# from test_layer_norm_op import *
import sys
import unittest
from functools import reduce
from operator import mul

sys.path.append("../../mkldnn")
import numpy as np
from op_test import _set_use_system_allocator, convert_float_to_uint16
from test_layer_norm_mkldnn_op_deprecated import (
    TestLayerNormMKLDNNOp,
    _reference_layer_norm_naive,
)
from utils import pir_executor_guard

import paddle
from paddle import base, enable_static
from paddle.base import core

np.random.random(123)

_set_use_system_allocator(True)


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestLayerNormBF16MKLDNNOp(TestLayerNormMKLDNNOp):
    def __assert_close(self, tensor, np_array, msg, rtol=2e-02, atol=2):
        np.testing.assert_allclose(
            np.array(tensor), np_array, rtol=rtol, atol=atol, err_msg=msg
        )

    def check_forward(
        self, shape, begin_norm_axis, with_scale_bias=True, with_is_test=False
    ):
        # attr
        epsilon = 0.00001
        x_shape = shape
        D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
        scale_shape = [D]

        np.random.seed(123)
        x = np.random.random_sample(x_shape).astype(np.float32)
        x_bf16 = convert_float_to_uint16(x)

        if with_scale_bias:
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
        else:
            scale = np.array([])
            bias = np.array([])

        # reference forward & backward
        y, mean, variance = _reference_layer_norm_naive(
            x, scale, bias, epsilon, begin_norm_axis
        )

        y_bf16 = convert_float_to_uint16(y)

        var_dict = locals()
        var_names = ['x_bf16', 'mean', 'variance', 'y_bf16']
        if with_scale_bias:
            var_names.append('scale')
            var_names.append('bias')
        ground_truth = {name: var_dict[name] for name in var_names}
        with paddle.pir_utils.OldIrGuard():
            program = base.Program()
            with base.program_guard(program):
                block = program.global_block()

                # scale and bias are fp32 and other vars are of bf16
                for name in ground_truth:
                    if name == 'x_bf16' or name == 'y_bf16':
                        block.create_var(
                            name=name,
                            dtype='uint16',
                            shape=ground_truth[name].shape,
                        )
                    else:
                        block.create_var(
                            name=name,
                            dtype='float32',
                            shape=ground_truth[name].shape,
                        )

                inputs = {"X": block.var('x_bf16')}
                if with_scale_bias:
                    inputs["Scale"] = block.var('scale')
                    inputs["Bias"] = block.var('bias')

                block.append_op(
                    type="layer_norm",
                    inputs=inputs,
                    outputs={
                        "Y": block.var('y_bf16'),
                        "Mean": block.var('mean'),  # share the same memory
                        "Variance": block.var(
                            'variance'
                        ),  # share the same memory
                    },
                    attrs={
                        "epsilon": epsilon,
                        "begin_norm_axis": begin_norm_axis,
                        "use_mkldnn": True,
                        "is_test": with_is_test,
                    },
                )

                exe = base.Executor(core.CPUPlace())

                input_list = ['x_bf16']
                if with_scale_bias:
                    input_list.append('scale')
                    input_list.append('bias')

                out = exe.run(
                    program,
                    feed={name: var_dict[name] for name in input_list},
                    fetch_list=['y_bf16', 'mean', 'variance'],
                )
                self.__assert_close(y_bf16, out[0], "y_bf16", 2)
                if not with_is_test:
                    self.__assert_close(mean, out[1], "mean")
                    self.__assert_close(variance, out[2], "variance", 1e-3)

    def test_check_forward_with_is_test(self):
        with pir_executor_guard():
            self.check_forward(
                shape=[2, 3, 4, 5], begin_norm_axis=3, with_is_test=True
            )

    # TODO (jczaja): Enable those to test when enabling training using bf16
    def test_check_forward_with_scale_and_bias(self):
        pass

    def test_check_forward_without_scale_and_bias(self):
        pass


if __name__ == "__main__":
    enable_static()
    unittest.main()
