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


class TestConv2dAddBf16Pass(PassTest):
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
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                # out = paddle.add(conv2d(x), bias)
                out = conv2d(x)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.conv2d": 1,
                    "pd_op.conv2d": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestMultiplyOpAddBf16Pass(PassTest):
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

                residual = paddle.static.data(
                    name="residual", shape=[5], dtype='float32'
                )
                out = paddle.multiply(x, y)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "residual": np.random.random(5).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.multiply": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestFcBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
                w = paddle.static.data(name='w', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name="y", shape=[3], dtype='float32')
                out = paddle.add(paddle.matmul(x, w), y)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'matmul_add_act_fuse_pass': {}},
                    {'fc_onednn_enable_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((3, 2)).astype("float32"),
                    "w": np.random.random((2, 3)).astype("float32"),
                    "y": np.random.random(3).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fc": 1,
                    "pd_op.matmul": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestLayerNormBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x1 = paddle.static.data(
                    name='x1', shape=[1, 30], dtype='float32'
                )

                layer_norm = paddle.nn.LayerNorm(x1.shape[-1:])
                out = layer_norm(x1)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x1": np.random.random((1, 30)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.layer_norm": 1,
                    "pd_op.layer_norm": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestMultiplyAddOpAddBf16Pass(PassTest):
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

                residual = paddle.static.data(
                    name="residual", shape=[5], dtype='float32'
                )
                multiply_out = paddle.multiply(x, y)
                out = paddle.add(multiply_out, residual)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "residual": np.random.random(5).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.multiply": 1,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestAddReluBf16Pass(PassTest):
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
                act_op = paddle.nn.ReLU()
                add = paddle.add(x, y)
                out = act_op(add)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]

                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.relu": 0,
                    "pd_op.add": 0,
                    "onednn_op.add": 1,
                    "onednn_op.relu": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestAddsigmoidBf16Pass(PassTest):
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
                # act_op = paddle.nn.ReLU()
                add = paddle.add(x, y)
                # out = act_op(add)
                out = paddle.nn.functional.sigmoid(add)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]

                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.sigmoid": 0,
                    "pd_op.add": 0,
                    "onednn_op.add": 1,
                    "onednn_op.sigmoid": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestfusedConv2dFusePass(PassTest):
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
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                out = paddle.add(conv2d(x), bias)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'conv2d_bias_fuse_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.fused_conv2d": 1,
                    "pd_op.conv2d": 0,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestReshapeBf16Pass(PassTest):
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
                reshape_x = paddle.reshape(x, [5, 125])
                reshape_y = paddle.reshape(y, [125, 5])

                out = paddle.matmul(reshape_x, reshape_y)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.matmul": 1,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestPool2dBf16Pass(PassTest):
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
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2D(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                # out = paddle.add(conv2d(x), bias)
                conv2d_ = conv2d(x)
                out = paddle.nn.functional.max_pool2d(
                    conv2d_, kernel_size=2, stride=2
                )
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.conv2d": 1,
                    # "onednn_op.add": 1,
                    "pd_op.conv2d": 0,
                    "pd_op.add": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestSumOpBf16Pass(PassTest):
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
                residual = paddle.static.data(
                    name="residual", shape=[5], dtype='float32'
                )
                matmul_out = paddle.matmul(x, y)
                out = paddle.sum(matmul_out, axis=-1)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "residual": np.random.random(5).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.matmul": 1,
                    "onednn_op.sum": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestAddClipBf16Pass(PassTest):
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
                add = paddle.add(x, y)
                out = paddle.clip(add, -5, 5)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]

                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.clip": 0,
                    "pd_op.add": 0,
                    "onednn_op.add": 1,
                    "onednn_op.clip": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestSliceOpBf16Pass(PassTest):
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
                add = paddle.add(x, y)
                out = paddle.slice(add, axes=[1], starts=[0], ends=[1])
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]

                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.slice": 0,
                    "pd_op.add": 0,
                    "onednn_op.add": 1,
                    "onednn_op.slice": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestSqueezeOpBf16Pass(PassTest):
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
                add = paddle.add(x, y)
                out = paddle.squeeze(add, axis=1)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]

                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.squeeze": 0,
                    "pd_op.add": 0,
                    "onednn_op.add": 1,
                    "onednn_op.squeeze": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestpreluBf16Pass(PassTest):
    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                x = paddle.static.data(
                    name='x', shape=[2, 3, 4, 5], dtype='float32'
                )
                out = paddle.nn.PReLU(3)(x)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]

                self.feeds = {
                    "x": np.random.random((2, 3, 4, 5)).astype("float32"),
                }

                self.fetch_list = [out]
                self.valid_op_map = {
                    "pd_op.prelu": 0,
                    "onednn_op.prelu": 1,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestSqueezeBf16Pass(PassTest):
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

                out = paddle.squeeze(x, axis=1)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.squeeze": 1,
                    "pd_op.squeeze": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestClipBf16Pass(PassTest):
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

                out = paddle.clip(x)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.clip": 1,
                    "pd_op.clip": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestsoftmaxBf16Pass(PassTest):
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

                out = paddle.nn.functional.softmax(x)
                out = paddle.assign(out)
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "y": np.random.random((5, 5, 5, 5)).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.softmax": 1,
                    "pd_op.softmax": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


class TestConvTransposeOpBf16Pass(PassTest):
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
                bias_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                bias = paddle.static.create_parameter(
                    shape=[1], dtype='float32', attr=bias_attr, is_bias=False
                )
                w_attr = paddle.ParamAttr(
                    learning_rate=0.0,
                    initializer=paddle.nn.initializer.Normal(mean=0.0, std=2.0),
                )
                conv2d = paddle.nn.Conv2DTranspose(
                    in_channels=5,
                    out_channels=1,
                    kernel_size=[1, 1],
                    groups=1,
                    stride=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    data_format='NCHW',
                    bias_attr=False,
                    weight_attr=w_attr,
                )

                out = paddle.assign(conv2d(x))
                self.pass_attr_list = [
                    {'onednn_placement_pass': {}},
                    {'cpu_bfloat16_placement_pass': {}},
                    {'cpu_bfloat16_pass': {}},
                    {'cpu_bfloat16_type_placement_pass': {}},
                ]
                self.feeds = {
                    "x": np.random.random((5, 5, 5, 5)).astype("float32"),
                    "bias": np.random.random(1).astype("float32"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {
                    "onednn_op.conv2d_transpose": 1,
                    "pd_op.conv2d_transpose": 0,
                }
                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        self.places.append(paddle.CPUPlace())
        self.skip_accuracy_verification = True

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
