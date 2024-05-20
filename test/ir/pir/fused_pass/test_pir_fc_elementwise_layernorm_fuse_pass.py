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

import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle.base import core

paddle.enable_static()


class TestFcElementwiseLayerNormFusePattern(PassTest):
    r"""
    fc     Y1
     \     /
       Add
        |
      LayerNorm
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for x_shape in [[3, 2]]:
            for w_shape in [[2, 3]]:
                for y_shape in [[1, 3], [3]]:
                    for bias_shape in [[3, 3]]:
                        for with_relu in [True, False]:
                            with paddle.pir_utils.IrGuard():
                                start_prog = paddle.static.Program()
                                main_prog = paddle.static.Program()
                                with paddle.pir.core.program_guard(
                                    main_prog, start_prog
                                ):
                                    x = paddle.static.data(
                                        name='x', shape=x_shape, dtype='float32'
                                    )
                                    w = paddle.static.data(
                                        name='w', shape=w_shape, dtype='float32'
                                    )
                                    y = paddle.static.data(
                                        name='y', shape=y_shape, dtype='float32'
                                    )
                                    if with_relu:
                                        relu_op = paddle.nn.ReLU()
                                        fc_out = relu_op(
                                            paddle.add(paddle.matmul(x, w), y)
                                        )
                                    else:
                                        fc_out = paddle.add(
                                            paddle.matmul(x, w), y
                                        )

                                    bias1 = paddle.static.data(
                                        name='bias1',
                                        shape=bias_shape,
                                        dtype='float32',
                                    )

                                    add_out = paddle.add(fc_out, bias1)
                                    layer_norm = paddle.nn.LayerNorm(
                                        add_out.shape[-1:]
                                    )
                                    out = layer_norm(add_out)
                                    out = paddle.assign(out)
                                    self.pass_attr_list.append(
                                        {'matmul_add_act_fuse_pass': {}}
                                    )
                                    self.pass_attr_list.append(
                                        {
                                            'fc_elementwise_layernorm_fuse_pass': {}
                                        }
                                    )
                                    self.feeds = {
                                        "x": np.random.random(x_shape).astype(
                                            "float32"
                                        ),
                                        "w": np.random.random(w_shape).astype(
                                            "float32"
                                        ),
                                        "y": np.random.random(y_shape).astype(
                                            "float32"
                                        ),
                                        "bias1": np.random.random(
                                            bias_shape
                                        ).astype("float32"),
                                    }
                                    self.fetch_list = [out]
                                    self.valid_op_map = {
                                        "pd_op.add": 0,
                                        "pd_op.relu": 0,
                                        "pd_op.matmul": 0,
                                        "pd_op.fused_fc_elementwise_layernorm": 1,
                                    }

                                    yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
