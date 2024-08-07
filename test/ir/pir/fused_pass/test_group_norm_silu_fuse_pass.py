# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from pass_test import PassTest

import paddle
from paddle.base import core
from paddle.pir.core import create_parameter


class GroupNormSiluPattern(PassTest):
    r"""
    group_norm
        |
      silu
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[2, 6, 4, 2]]:
            dtype = None
            if core.is_compiled_with_xpu():
                dtype = 'float32'
            elif core.is_compiled_with_cuda():
                dtype = 'float16'
            for epilson in [1e-5]:
                for groups in [2]:
                    rand_value = (
                        0.001
                        * paddle.rand(shape=[x_shape[1]], dtype=dtype).numpy()
                    )
                    with paddle.pir_utils.IrGuard():
                        start_prog = paddle.static.Program()
                        main_prog = paddle.static.Program()
                        with paddle.pir.core.program_guard(
                            main_prog, start_prog
                        ):
                            x = paddle.static.data(
                                name='x', shape=x_shape, dtype=dtype
                            )
                            w = create_parameter(
                                shape=[x_shape[1]],
                                dtype=dtype,
                                initializer=paddle.nn.initializer.Assign(
                                    rand_value
                                ),
                            )
                            b = create_parameter(
                                shape=[x_shape[1]],
                                dtype=dtype,
                                initializer=paddle.nn.initializer.Assign(
                                    rand_value
                                ),
                            )
                            group_norm_out = paddle.nn.functional.group_norm(
                                x,
                                num_groups=groups,
                                epsilon=epilson,
                                weight=w,
                                bias=b,
                            )
                            out = paddle.nn.functional.silu(group_norm_out)
                            out = paddle.assign(out)
                            if core.is_compiled_with_xpu():
                                self.pass_attr_list = [
                                    {'group_norm_silu_fuse_pass': {}},
                                ]
                            elif core.is_compiled_with_cuda():
                                self.pass_attr_list = [
                                    {'group_norm_silu_fuse_pass': {}},
                                    {'transfer_layout_pass': {}},
                                ]
                            self.feeds = {
                                "x": np.random.random(x_shape).astype(dtype),
                            }
                            self.fetch_list = [out]
                            if core.is_compiled_with_xpu():
                                self.valid_op_map = {
                                    "pd_op.silu": 0,
                                    "pd_op.group_norm": 0,
                                    "pd_op.group_norm_silu_xpu": 1,
                                }
                            elif core.is_compiled_with_cuda():
                                self.valid_op_map = {
                                    "pd_op.silu": 0,
                                    "pd_op.group_norm": 0,
                                    "pd_op.add_group_norm_silu": 1,
                                }

                            yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_xpu():
            self.places.append(paddle.XPUPlace(0))
        elif core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
