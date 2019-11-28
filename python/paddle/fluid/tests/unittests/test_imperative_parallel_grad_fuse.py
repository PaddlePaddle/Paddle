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

import contextlib
import copy
import unittest
import numpy as np
from collections import OrderedDict

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.dygraph.parallel import DataParallel
from paddle.fluid.dygraph.base import to_variable, no_grad
from test_imperative_mnist import MNIST


class DataParallelGradFuseTest(DataParallel):
    def __init__(self, layers):
        strategy = core.ParallelStrategy()
        super(DataParallelGradFuseTest, self).__init__(layers, strategy)

    @no_grad
    def grad_coalesce_and_split(self):
        # Get grad vars
        grad_var_set = set()
        grad_vars = []
        for param in self._layers.parameters():
            if param.trainable and param._ivar._grad_ivar():
                g_var = fluid.framework.Variable(
                    block=self._helper.main_program.current_block(),
                    name=param._ivar._grad_name(),
                    stop_gradient=True,
                    ivar=param._ivar._grad_ivar())
                grad_vars.append(g_var)
                assert g_var not in grad_var_set
                grad_var_set.add(g_var)
        # Group construct
        mega_bytes = 128 * 1024 * 1024
        group_idx = 0
        memory_counter = 0
        grad_var_groups = OrderedDict()
        dtype = grad_vars[0].dtype
        for g_var in grad_vars:
            bytes = np.prod(g_var.shape) * core.size_of_dtype(g_var.dtype)
            if memory_counter < mega_bytes and dtype == g_var.dtype:
                memory_counter += bytes
            else:
                memory_counter = bytes
                group_idx += 1
            grad_var_groups.setdefault(group_idx, []).append(g_var)
        # copy gradients shape to compare
        orig_grad_shapes = []
        for g_var in grad_vars:
            orig_grad_shapes.append(g_var.shape)
        # coalesce and split
        coalesced_grads_and_vars = self._coalesce_tensors(grad_var_groups)
        self._split_tensors(coalesced_grads_and_vars)
        return orig_grad_shapes, grad_vars


class TestImperativeParallelGradFuse(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64

    def test_coalesce_split(self):
        with fluid.dygraph.guard():
            mnist = MNIST("mnist")
            mnist = DataParallelGradFuseTest(mnist)
            opt = fluid.optimizer.Adam(learning_rate=0.001)

            train_reader = paddle.batch(
                paddle.dataset.mnist.train(),
                batch_size=self.batch_size,
                drop_last=True)

            for data in train_reader():
                x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    self.batch_size, 1)

                img = to_variable(x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost = mnist(img)
                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()

                orig_grad_shapes, grad_vars = mnist.grad_coalesce_and_split()
                for orig_g_shape, g_var in zip(orig_grad_shapes, grad_vars):
                    self.assertEqual(orig_g_shape, g_var.shape)

                opt.minimize(avg_loss)
                mnist.clear_gradients()

    def test_reshape_inplace(self):
        with fluid.dygraph.guard():
            mnist = MNIST("mnist")
            mnist = DataParallelGradFuseTest(mnist)

            ori_shape = (2, 25)
            new_shape = (5, 10)
            x = np.random.random(ori_shape).astype("float32")
            mnist._reshape_inplace(x, new_shape)
            self.assertEqual(x.shape, new_shape)


if __name__ == '__main__':
    unittest.main()
