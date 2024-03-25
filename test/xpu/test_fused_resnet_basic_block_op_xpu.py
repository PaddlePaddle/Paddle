#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import OpTest

import paddle
from paddle import base, nn
from paddle.base import core
from paddle.base.framework import default_main_program
from paddle.incubate.xpu.resnet_block import ResNetBasicBlock


class XPUTestResNetBasicBlockOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "resnet_basic_block"
        self.use_dynamic_create_class = False

    class TestResNetBasicBlockOp(OpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.__class__.op_type = "resnet_basic_block"
            self.__class__.no_need_check_grad = True
            self.getShape()
            self.getDiff()
            self.getShortcut()
            paddle.set_default_dtype(self.dtype)

            self.src = np.random.random(self.input_size).astype(self.dtype)
            self.dout = np.random.random(self.output_size).astype(self.dtype)

        def getShape(self):
            self.in_channels = 8
            self.out_channels = 8
            self.stride = 1
            self.input_size = [2, 8, 32, 32]  # NCHW
            self.output_size = [2, 8, 32, 32]  # NCHW

        def getDiff(self):
            self.rtol = 1e-3
            self.atol = 1e-3

        def getShortcut(self):
            self.has_shortcut = False

        def Base(self):
            conv1_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal(),
                learning_rate=0.001,
            )
            conv2_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal(),
                learning_rate=0.001,
            )
            conv3_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal(),
                learning_rate=0.001,
            )
            bn1_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            )
            bn1_bias = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
            bn2_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            )
            bn2_bias = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
            bn3_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            )
            bn3_bias = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )

            self.conv1 = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                weight_attr=conv1_weight,
                bias_attr=None,
                data_format='NCHW',
            )
            self.bn1 = paddle.nn.BatchNorm(
                self.out_channels,
                act='relu',
                param_attr=bn1_weight,
                bias_attr=bn1_bias,
                data_layout='NCHW',
            )
            self.conv2 = nn.Conv2D(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=conv2_weight,
                bias_attr=None,
                data_format='NCHW',
            )
            self.bn2 = paddle.nn.BatchNorm(
                self.out_channels,
                act=None,
                param_attr=bn2_weight,
                bias_attr=bn2_bias,
                data_layout='NCHW',
            )
            self.conv3 = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                weight_attr=conv3_weight,
                bias_attr=None,
                data_format='NCHW',
            )
            self.bn3 = paddle.nn.BatchNorm(
                self.out_channels,
                act=None,
                param_attr=bn3_weight,
                bias_attr=bn3_bias,
                data_layout='NCHW',
            )
            self.relu = nn.ReLU()

            tensor_src = paddle.to_tensor(self.src, stop_gradient=False)
            if self.has_shortcut:
                z_out = self.bn3(self.conv3(tensor_src))
            else:
                z_out = tensor_src
            bn1_out = self.bn1(self.conv1(tensor_src))
            bn2_out = self.bn2(self.conv2(bn1_out))
            result = self.relu(bn2_out + z_out)
            paddle.autograd.backward(
                [result], [paddle.to_tensor(self.dout)], True
            )
            return result, tensor_src.grad

        def FusedResNetBasicBlock(self):
            fused_conv1_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal(),
                learning_rate=0.001,
            )
            fused_conv2_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal(),
                learning_rate=0.001,
            )
            fused_conv3_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal(),
                learning_rate=0.001,
            )
            fused_bn1_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            )
            fused_bn1_bias = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
            fused_bn2_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            )
            fused_bn2_bias = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )
            fused_bn3_weight = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)
            )
            fused_bn3_bias = base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0)
            )

            if self.has_shortcut:
                self.resnet_basic_block = ResNetBasicBlock(
                    num_channels1=self.in_channels,
                    num_filter1=self.out_channels,
                    filter1_size=3,
                    num_channels2=self.out_channels,
                    num_filter2=self.out_channels,
                    filter2_size=3,
                    num_channels3=self.in_channels,
                    num_filter3=self.out_channels,
                    filter3_size=1,
                    filter1_attr=fused_conv1_weight,
                    scale1_attr=fused_bn1_weight,
                    bias1_attr=fused_bn1_bias,
                    filter2_attr=fused_conv2_weight,
                    scale2_attr=fused_bn2_weight,
                    bias2_attr=fused_bn2_bias,
                    filter3_attr=fused_conv3_weight,
                    scale3_attr=fused_bn3_weight,
                    bias3_attr=fused_bn3_bias,
                    stride1=self.stride,
                    stride2=1,
                    stride3=self.stride,
                    act='relu',
                    padding1=1,
                    padding2=1,
                    padding3=0,
                    has_shortcut=True,
                )
            else:
                self.resnet_basic_block = ResNetBasicBlock(
                    num_channels1=self.in_channels,
                    num_filter1=self.out_channels,
                    filter1_size=3,
                    num_channels2=self.out_channels,
                    num_filter2=self.out_channels,
                    filter2_size=3,
                    num_channels3=self.in_channels,
                    num_filter3=self.out_channels,
                    filter3_size=1,
                    filter1_attr=fused_conv1_weight,
                    scale1_attr=fused_bn1_weight,
                    bias1_attr=fused_bn1_bias,
                    filter2_attr=fused_conv2_weight,
                    scale2_attr=fused_bn2_weight,
                    bias2_attr=fused_bn2_bias,
                    filter3_attr=fused_conv3_weight,
                    scale3_attr=fused_bn3_weight,
                    bias3_attr=fused_bn3_bias,
                    stride1=self.stride,
                    stride2=1,
                    stride3=self.stride,
                    act='relu',
                    padding1=1,
                    padding2=1,
                    padding3=1,
                    has_shortcut=False,
                )

            x = paddle.to_tensor(self.src, stop_gradient=False)
            out = self.resnet_basic_block.forward(x)
            paddle.autograd.backward([out], [paddle.to_tensor(self.dout)])
            return out, x.grad

        def test_out_and_grad_has_shortcut(self):
            self.has_shortcut = True
            default_main_program().random_seed = 1
            base_out, base_grad = self.Base()
            fused_out, fused_grad = self.FusedResNetBasicBlock()
            np.testing.assert_allclose(
                base_out.numpy(),
                fused_out.numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                base_grad.numpy(),
                fused_grad.numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )

        def test_out_and_grad(self):
            self.has_shortcut = False
            default_main_program().random_seed = 1
            base_out, base_grad = self.Base()
            fused_out, fused_grad = self.FusedResNetBasicBlock()
            np.testing.assert_allclose(
                base_out.numpy(),
                fused_out.numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                base_grad.numpy(),
                fused_grad.numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )


support_types = get_xpu_op_support_types('resnet_basic_block')
for stype in support_types:
    create_test_class(
        globals(),
        XPUTestResNetBasicBlockOp,
        stype,
        ignore_device_version=[core.XPUVersion.XPU1],
    )

if __name__ == '__main__':
    unittest.main()
