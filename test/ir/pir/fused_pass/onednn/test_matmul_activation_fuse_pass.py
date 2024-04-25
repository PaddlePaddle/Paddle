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


class TestMatmulActFusePatternCase1(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      relu
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.relu(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.relu": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase2(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
     swish
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.swish(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.swish": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase3(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      tanh
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.abs(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.abs": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulClipFusePatternCase4(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      clip
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.clip(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.clip": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase5(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      gelu
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.gelu(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.gelu": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase6(PassTest):
    r'''
      x     y
       \   /
       matmul
         |
    hardsigmoid
         |
        out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.hardsigmoid(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.hardsigmoid": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase7(PassTest):
    r'''
     x     y
      \   /
      matmul
        |
    hardswish
        |
       out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.hardswish(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.hardswish": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase8(PassTest):
    r'''
     x     y
      \   /
      matmul
        |
    leaky_relu
        |
       out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.leaky_relu(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.leaky_relu": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase9(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      mish
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.mish(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.mish": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase10(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
     relu6
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.relu6(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.relu6": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase11(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
    sigmoid
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.sigmoid(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.sigmoid": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase12(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      sqrt
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.sqrt(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.sqrt": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulAddFusePatternCase13(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      tanh
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.tanh(matmul_out)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.tanh": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestFusedMatmulActFusePattern(PassTest):
    r'''
    x     y
     \   /
     matmul  resdual(data)
        \   /
         add
          |
         relu
          |
         out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                bias = paddle.static.data(
                    name="bias", shape=[1], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.add(matmul_out, bias)
                act_out = paddle.nn.functional.relu(out)
                act_out = paddle.assign(act_out)
                self.pass_attr_list = [
                    {'matmul_elementwise_add_fuse_pass': {}},
                    {'matmul_activation_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [act_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.add": 0,
                    "pd_op.relu": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestFusedMatmulClipFusePattern(PassTest):
    r'''
    x     y
     \   /
     matmul  resdual(data)
        \   /
         add
          |
         clip
          |
         out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                bias = paddle.static.data(
                    name="bias", shape=[1], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.add(matmul_out, bias)
                act_out = paddle.clip(out)
                act_out = paddle.assign(act_out)
                self.pass_attr_list = [
                    {'matmul_elementwise_add_fuse_pass': {}},
                    {'matmul_activation_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [act_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.add": 0,
                    "pd_op.clip": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestFusedMatmulsigmoidFusePattern(PassTest):
    r'''
    x     y
     \   /
     matmul  resdual(data)
        \   /
         add
          |
     hardsigmoid
          |
         out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                bias = paddle.static.data(
                    name="bias", shape=[1], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.add(matmul_out, bias)
                act_out = paddle.nn.functional.hardsigmoid(out)
                act_out = paddle.assign(act_out)
                self.pass_attr_list = [
                    {'matmul_elementwise_add_fuse_pass': {}},
                    {'matmul_activation_fuse_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [act_out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.add": 0,
                    "pd_op.hardsigmoid": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())

    def test_check_output(self):
        self.check_pass_correct()


class TestMatmulGeluTanhFusePatternCase14(PassTest):
    r'''
    x     y
     \   /
     matmul
       |
      gelu
       |
      out
    '''

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[5, 5, 5, 5], dtype='float32'
                )
                y = paddle.static.data(
                    name='y', shape=[5, 5, 5, 5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.nn.functional.gelu(matmul_out, approximate=True)
                out = paddle.assign(out)
                self.pass_attr_list = [{'matmul_activation_fuse_pass': {}}]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_matmul": 1,
                    "pd_op.matmul": 0,
                    "pd_op.gelu": 0,
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
