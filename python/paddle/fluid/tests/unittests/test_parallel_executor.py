#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle.fluid as fluid
from parallel_executor_test_base import TestParallelExecutorBase
import numpy as np
import unittest

def simple_fc_net(use_feed):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = fluid.layers.fc(
        img,
        size=200,
        act='tanh',
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss

class TestParallelExecutorStrategy(TestParallelExecutorBase):
    def _init_data(self):
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        return img, label

    def _run_options(self,
                     memory_opt=True,
                     batch_size=None,
                     allow_op_delay=False,
                     use_reduce=False,
                     use_ir_memory_optimize=True,
                     enable_inplace=True,
                     fuse_elewise_add_act_ops=False,
                     fuse_all_optimizer_ops=False,
                     fuse_all_reduce_ops=False,
                     fuse_relu_depthwise_conv=False,
                     use_fast_executor=False,
                     enable_sequential_execution=False):

        for use_cuda in [True, False]:
            if use_cuda and not fluid.core.is_compiled_with_cuda():
                    return

            img, label = self._init_data()
            self.check_network_convergence(
                simple_fc_net,
                feed_dict={"image": img,
                           "label": label},
                use_cuda=use_cuda,
                memory_opt=memory_opt,
                batch_size=batch_size,
                allow_op_delay=allow_op_delay,
                use_parallel_executor=True,
                use_reduce=use_reduce,
                use_ir_memory_optimize=use_ir_memory_optimize,
                enable_inplace=enable_inplace,
                fuse_elewise_add_act_ops=fuse_elewise_add_act_ops,
                fuse_all_optimizer_ops=fuse_all_optimizer_ops,
                fuse_all_reduce_ops=fuse_all_reduce_ops,
                fuse_relu_depthwise_conv=fuse_relu_depthwise_conv,
                use_fast_executor=use_fast_executor,
                enable_sequential_execution=enable_sequential_execution)

    def test_memory_opt(self):
        for memory_opt in [True, False]:
            self._run_options(memory_opt=memory_opt)

    def test_allow_op_delay(self):
        for allow_op_delay in [True, False]:
            self._run_options(allow_op_delay=allow_op_delay)

    def test_use_reduce(self):
        for use_reduce in [True, False]:
            self._run_options(use_reduce=use_reduce)

    def test_use_ir_memory_optimize(self):
        for use_ir_memory_optimize in [True, False]:
            self._run_options(use_ir_memory_optimize=use_ir_memory_optimize)

    def test_enable_inplace(self):
        for enable_inplace in [True, False]:
            self._run_options(enable_inplace=enable_inplace)

    def test_fuse_elewise_add_act_ops(self):
        for fuse_elewise_add_act_ops in [True, False]:
            self._run_options(fuse_elewise_add_act_ops=fuse_elewise_add_act_ops)

    def test_fuse_all_optimizer_ops(self):
        for fuse_all_optimizer_ops in [True, False]:
            self._run_options(fuse_all_optimizer_ops=fuse_all_optimizer_ops)

    def test_fuse_all_reduce_ops(self):
        for fuse_all_reduce_ops in [True, False]:
            self._run_options(fuse_all_reduce_ops=fuse_all_reduce_ops)

    def test_fuse_relu_depthwise_conv(self):
        for fuse_relu_depthwise_conv in [True, False]:
            self._run_options(fuse_relu_depthwise_conv=fuse_relu_depthwise_conv)

    def test_use_fast_executor(self):
        for use_fast_executor in [True, False]:
            self._run_options(use_fast_executor=use_fast_executor)

    def test_enable_sequential_execution(self):
        for enable_sequential_execution in [True, False]:
            self._run_options(enable_sequential_execution=enable_sequential_execution)


if __name__ == '__main__':
    unittest.main()
