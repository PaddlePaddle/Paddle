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

paddle.enable_static()


class TestFusedTransposeReshapeFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[4, 16, 1, 32], dtype='float32'
                )

                squeeze_out = paddle.squeeze(x, axis=[2])
                transpose = paddle.transpose(squeeze_out, [0, 1, 2])
                out = paddle.reshape(
                    x=transpose, shape=[np.prod(transpose.shape)]
                )
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'squeeze_transpose_onednn_fuse_pass': {}},
                    {'operator_reshape_onednn_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((4, 16, 1, 32)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.transpose": 0,
                    "pd_op.reshape": 0,
                    "onednn_op.fused_transpose": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestTransposeReshapeFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 6, 4], dtype='float32'
                )

                # squeeze_out = paddle.squeeze(x, axis=[2])
                transpose = paddle.transpose(x, [0, 2, 1])
                out = paddle.reshape(
                    x=transpose, shape=[np.prod(transpose.shape)]
                )
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'operator_reshape_onednn_fuse_pass': {}}
                ]
                self.feeds = {
                    "x": np.random.random((2, 6, 4)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.transpose": 0,
                    "pd_op.reshape": 0,
                    "onednn_op.fused_transpose": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestFcReshapeFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for x_shape in [[3, 2]]:
            for w_shape in [[2, 3]]:
                for y_shape in [[3], [1, 3]]:
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
                            fc = paddle.add(paddle.matmul(x, w), y)
                            out = paddle.reshape(
                                x=fc, shape=[np.prod(fc.shape)]
                            )
                            out = paddle.assign(out)
                            self.pass_attr_list = [
                                {'matmul_add_act_fuse_pass': {}},
                                {"fc_onednn_enable_pass": {}},
                                {"operator_reshape_onednn_fuse_pass": {}},
                            ]
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
                            }
                            self.fetch_list = [out]
                            self.valid_op_map = {
                                "pd_op.add": 0,
                                "pd_op.matmul": 0,
                                "pd_op.fc": 0,
                                "pd_op.scale": 0,
                                "onednn_op.fc": 1,
                            }

                            return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestFcReshapeTransposeFusePass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for x_shape in [[8, 2]]:
            for w_shape in [[2, 3]]:
                for y_shape in [[3], [1, 3]]:
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
                            fc = paddle.add(paddle.matmul(x, w), y)
                            out = paddle.reshape(x=fc, shape=[6, 2, 2])
                            transpose = paddle.transpose(out, [0, 2, 1])
                            out = paddle.assign(transpose)
                            self.pass_attr_list = [
                                {'matmul_add_act_fuse_pass': {}},
                                {"fc_onednn_enable_pass": {}},
                                {'operator_reshape_onednn_fuse_pass': {}},
                            ]
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
                            }
                            self.fetch_list = [out]
                            self.valid_op_map = {
                                "pd_op.add": 0,
                                "pd_op.matmul": 0,
                                "pd_op.fc": 0,
                                "pd_op.scale": 0,
                                "pd_op.reshape": 0,
                                "onednn_op.fc": 1,
                                "pd_op.transpose": 1,
                            }

                            return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
