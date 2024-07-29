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

import os
import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle.base import core

paddle.enable_static()


class MatmulHorizontalFusePattern(PassTest):
    r"""
    x_var   q_var   k_var   v_var
      \       |       |       /
        matmul
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[3, 2]]:
            for q_shape in [[2, 2]]:
                for k_shape in [[2, 2]]:
                    for v_shape in [[2, 2]]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(main_prog, start_prog):
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float32'
                                )
                                q = paddle.static.data(
                                    name='q', shape=q_shape, dtype='float32'
                                )
                                k = paddle.static.data(
                                    name='k', shape=k_shape, dtype='float32'
                                )
                                v = paddle.static.data(
                                    name='v', shape=v_shape, dtype='float32'
                                )
                                print(x)
                                out_q = paddle.matmul(x, q)
                                out_k = paddle.matmul(x, k)
                                out_v = paddle.matmul(x, v)
                                print(out_q)
                                out_1 = paddle.assign(out_q)
                                out_2 = paddle.assign(out_k)
                                out_3 = paddle.assign(out_v)
                                print("hahahhahahhahahhahahhahahhahha")
                                self.pass_attr_list = [
                                   
                                    {'matmul_horizontal_fuse_pass': {}}
                                ]
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float32"
                                    ),
                                    "q": np.random.random(q_shape).astype(
                                        "float32"
                                    ),
                                    "k": np.random.random(k_shape).astype(
                                        "float32"
                                    ),
                                    "v": np.random.random(v_shape).astype(
                                        "float32"
                                    ),
                                }
                                self.fetch_list = [out_1, out_2, out_3]
                                # self.fetch_list = [out_1]
                                self.valid_op_map = {
                                    "pd_op.matmul": 1,
                                    # "pd_op.slice": 1,
                                    # "pd_op.concat": 1,
                                }
                                yield [main_prog, start_prog], False

    def setUp(self):
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()