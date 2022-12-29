# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


def fc_with_batchnorm(use_feed):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    hidden = img
    for _ in range(3):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)
            ),
        )

        hidden = paddle.static.nn.batch_norm(input=hidden)
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    loss = paddle.mean(loss)
    return loss


class TestIrInplace(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def _fc_with_batchnorm(self, ir_memory_optimize, enable_inplace):

        if not core.is_compiled_with_cuda():
            return
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        self.check_network_convergence(
            fc_with_batchnorm,
            feed_dict={"image": img, "label": label},
            use_device=DeviceType.CUDA,
            use_ir_memory_optimize=ir_memory_optimize,
            enable_inplace=enable_inplace,
        )

    def test_fc_with_batchnorm(self, delta=1e-3):
        loss00 = self._fc_with_batchnorm(False, False)
        loss10 = self._fc_with_batchnorm(True, False)
        loss01 = self._fc_with_batchnorm(False, True)
        loss11 = self._fc_with_batchnorm(True, True)
        self.assertAlmostEqual(loss00, loss10, delta=delta)
        self.assertAlmostEqual(loss00, loss01, delta=delta)
        self.assertAlmostEqual(loss00, loss11, delta=delta)


if __name__ == '__main__':
    unittest.main()
