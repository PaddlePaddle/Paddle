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

paddle.enable_static()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestSqueezeFcFusePattern(PassTest):
    r"""
    squeeze
        \

        Matmul     Y
           \     /
              Add
               |
              Relu
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for y_shape in [[128], [1, 128]]:
            for w_shape in [[128, 128]]:
                pir_program = None
                with paddle.pir_utils.IrGuard():
                    pir_program = paddle.static.Program()
                    with paddle.pir.core.program_guard(pir_program):
                        x = paddle.static.data(
                            name='x', shape=[3, 128, 1, 1], dtype='float32'
                        )
                        w = paddle.static.data(
                            name='w', shape=w_shape, dtype='float32'
                        )
                        y = paddle.static.data(
                            name='y', shape=y_shape, dtype='float32'
                        )

                        out = paddle.add(
                            paddle.matmul(paddle.squeeze(x, [2, 3]), w), y
                        )

                        self.pass_list = ['fc_with_special_op_fuse_pass']
                        self.feeds = {
                            "x": np.random.random([3, 128, 1, 1]).astype(
                                "float32"
                            ),
                            "y": np.random.random(y_shape).astype("float32"),
                            "w": np.random.random(w_shape).astype("float32"),
                        }
                        self.fetch_list = [out]
                        self.valid_op_map = {
                            "pd_op.add": 0,
                            "pd_op.squeeze": 0,
                            "pd_op.matmul": 0,
                            "pd_op.fc": 1,
                        }

                        yield pir_program, False

    def setUp(self):
        self.place_runtime = "gpu"

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestReshapeFcFusePattern(PassTest):
    r"""
    reshape
        \

        Matmul     Y
           \     /
              Add
               |
              Relu
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for y_shape in [[192], [1, 192]]:
            pir_program = None
            with paddle.pir_utils.IrGuard():
                pir_program = paddle.static.Program()
                with paddle.pir.core.program_guard(pir_program):
                    x = paddle.static.data(
                        name='x', shape=[3, 144, 6, 32], dtype='float32'
                    )
                    w = paddle.static.data(
                        name='w', shape=[192, 192], dtype='float32'
                    )
                    y = paddle.static.data(
                        name='y', shape=y_shape, dtype='float32'
                    )

                    out = paddle.add(
                        paddle.matmul(paddle.reshape(x, [3, 144, -1]), w), y
                    )

                    self.pass_list = ['fc_with_special_op_fuse_pass']
                    self.feeds = {
                        "x": np.random.random([3, 255, 1, 1]).astype("float32"),
                        "w": np.random.random([255, 128]).astype("float32"),
                        "y": np.random.random(y_shape).astype("float32"),
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "pd_op.add": 0,
                        "pd_op.reshape": 0,
                        "pd_op.matmul": 0,
                        "pd_op.fc": 1,
                    }

                    yield pir_program, False

    def setUp(self):
        self.place_runtime = "gpu"

    def test_check_output(self):
        self.check_pass_correct()


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda(),
    "core is not complied with CUDA",
)
class TestFlattenFcFusePattern(PassTest):
    r"""
    flatten
        \

        Matmul     Y
           \     /
              Add
               |
              Relu
    """

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        for y_shape in [[128], [1, 128]]:
            pir_program = None
            with paddle.pir_utils.IrGuard():
                pir_program = paddle.static.Program()
                with paddle.pir.core.program_guard(pir_program):
                    x = paddle.static.data(
                        name='x', shape=[3, 255, 1, 1], dtype='float32'
                    )
                    w = paddle.static.data(
                        name='w', shape=[255, 128], dtype='float32'
                    )
                    y = paddle.static.data(
                        name='y', shape=y_shape, dtype='float32'
                    )

                    out = paddle.add(
                        paddle.matmul(paddle.flatten(x, start_axis=1), w), y
                    )

                    self.pass_list = ['fc_with_special_op_fuse_pass']
                    self.feeds = {
                        "x": np.random.random([3, 255, 1, 1]).astype("float32"),
                        "w": np.random.random([255, 128]).astype("float32"),
                        "y": np.random.random(y_shape).astype("float32"),
                    }
                    self.fetch_list = [out]
                    self.valid_op_map = {
                        "pd_op.add": 0,
                        "pd_op.flatten": 0,
                        "pd_op.matmul": 0,
                        "pd_op.fc": 1,
                    }

                    yield pir_program, False

    def setUp(self):
        self.place_runtime = "gpu"

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
