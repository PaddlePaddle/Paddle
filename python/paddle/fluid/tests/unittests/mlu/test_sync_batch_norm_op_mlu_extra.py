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
"""
test for sync bachnorm op.
for both FP32 and FP16 input.
"""

import unittest
import numpy as np
import os
import sys
import six
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.fluid import Program, program_guard

sys.path.append("..")
from op_test import OpTest, _set_use_system_allocator
from test_dist_base import TestDistBase

paddle.enable_static()


class TestDygraphSyncBatchNormAPIError(unittest.TestCase):

    def test_errors(self):
        if not core.is_compiled_with_mlu():
            return

        with program_guard(Program(), Program()):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10)
            x1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                         [[1, 1, 1, 1]], fluid.MLUPlace(0))
            self.assertRaises(TypeError, my_sync_batch_norm, x1)

            # the input dtype of SyncBatchNorm must be float16 or float32
            x2 = fluid.layers.data(name='x2', shape=[3, 4, 5, 6], dtype="int32")
            self.assertRaises(TypeError, my_sync_batch_norm, x2)


class TestConvertSyncBatchNorm(unittest.TestCase):

    def test_convert(self):
        if not core.is_compiled_with_mlu():
            return

        with program_guard(Program(), Program()):
            compare_model = paddle.nn.Sequential(paddle.nn.Conv2D(3, 5, 3),
                                                 paddle.nn.BatchNorm2D(5),
                                                 paddle.nn.BatchNorm2D(5))
            model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3), paddle.nn.BatchNorm2D(5),
                paddle.nn.BatchNorm2D(
                    5,
                    weight_attr=fluid.ParamAttr(name='bn.scale'),
                    bias_attr=fluid.ParamAttr(name='bn.bias')))
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            for idx, sublayer in enumerate(compare_model.sublayers()):
                if isinstance(sublayer, paddle.nn.BatchNorm2D):
                    self.assertEqual(
                        isinstance(model[idx], paddle.nn.SyncBatchNorm), True)


class TestConvertSyncBatchNormCast1(unittest.TestCase):

    def test_convert(self):
        if not core.is_compiled_with_mlu():
            return

        class Net(nn.Layer):

            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2D(3, 5, 3)
                self.bn = []
                bn = self.add_sublayer('bn', nn.BatchNorm2D(5))
                self.bn.append(bn)

            def forward(self, x):
                x = self.conv1(x)
                for bn in self.bn:
                    x = bn(x)
                return x

        model = nn.Sequential()
        model.add_sublayer('net1', Net())
        model.add_sublayer('net2', Net())
        compare_model = nn.Sequential()
        compare_model.add_sublayer('net1', Net())
        compare_model.add_sublayer('net2', Net())
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.assertEqual(len(compare_model.sublayers()), len(model.sublayers()))


class TestConvertSyncBatchNormCase2(unittest.TestCase):

    def test_convert(self):
        if not core.is_compiled_with_mlu():
            return

        with fluid.dygraph.guard(fluid.MLUPlace(0)):

            class SyBNNet(paddle.nn.Layer):

                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super(SyBNNet, self).__init__()
                    self.bn_s1 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(
                            out_ch,
                            weight_attr=paddle.ParamAttr(
                                regularizer=paddle.regularizer.L2Decay(0.))))
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC'))

                def forward(self, x):
                    x = self.bn_s1(x)
                    out = paddle.sum(paddle.abs(self.bn_s2(x)))
                    return out

            class BNNet(paddle.nn.Layer):

                def __init__(self, in_ch=3, out_ch=3, dirate=1):
                    super(BNNet, self).__init__()
                    self.bn_s1 = paddle.nn.BatchNorm3D(
                        out_ch,
                        weight_attr=paddle.ParamAttr(
                            regularizer=paddle.regularizer.L2Decay(0.)))
                    self.bn_s2 = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(
                        paddle.nn.BatchNorm3D(out_ch, data_format='NDHWC'))

                def forward(self, x):
                    x = self.bn_s1(x)
                    out = paddle.sum(paddle.abs(self.bn_s2(x)))
                    return out

            bn_model = BNNet()
            sybn_model = SyBNNet()
            np.random.seed(10)
            data = np.random.random([3, 3, 3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            bn_out = bn_model(x)
            sybn_out = sybn_model(x)
            np.testing.assert_allclose(
                bn_out.numpy(),
                sybn_out.numpy(),
                rtol=1e-05,
                err_msg='Output has diff. \n' + '\nBN     ' +
                str(bn_out.numpy()) + '\n' + 'Sync BN ' + str(sybn_out.numpy()))


class TestDygraphSyncBatchNormDataFormatError(unittest.TestCase):

    def test_errors(self):
        if not core.is_compiled_with_mlu():
            return

        with fluid.dygraph.guard(fluid.MLUPlace(0)):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10, data_format='CN')
            data = np.random.random([3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            self.assertRaises(ValueError, my_sync_batch_norm, x)


if __name__ == '__main__':
    unittest.main()
