# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import os
import sys

sys.path.append("..")

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.fluid import Program, program_guard

from paddle.fluid.tests.unittests.op_test import OpTest, _set_use_system_allocator

# _set_use_system_allocator(False)
paddle.enable_static()


class TestDygraphSyncBatchNormAPIError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10)
            x1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                         [[1, 1, 1, 1]], fluid.NPUPlace(0))
            self.assertRaises(TypeError, my_sync_batch_norm, x1)

            # the input dtype of SyncBatchNorm must be float16 or float32
            # float16 only can be set on GPU place and NPU place
            x2 = fluid.layers.data(name='x2', shape=[3, 4, 5, 6], dtype="int32")
            self.assertRaises(TypeError, my_sync_batch_norm, x2)


class TestConvertSyncBatchNorm(unittest.TestCase):

    def test_convert(self):
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


class TestDygraphSyncBatchNormDataFormatError(unittest.TestCase):

    def test_errors(self):
        with fluid.dygraph.guard(fluid.NPUPlace(0)):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10, data_format='CN')
            data = np.random.random([3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            self.assertRaises(ValueError, my_sync_batch_norm, x)


if __name__ == '__main__':
    unittest.main()
