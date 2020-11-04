# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest
import numpy as np
import six
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph.nn import Linear
import paddle.fluid.core as core
from paddle.fluid.optimizer import SGDOptimizer


class MLP(fluid.Layer):
    def __init__(self, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._linear1 = Linear(784, 10)
        self._linear2 = Linear(10, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        return y


class TestDataParallelBucket(unittest.TestCase):
    def create_varbase(self, dtype, shape,
                       type=core.VarDesc.VarType.LOD_TENSOR):
        return core.VarBase(dtype, shape, "", type, True)

    def test_construct_group0(self):
        # one dtype & one limit capability
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(
            self.create_varbase(core.VarDesc.VarType.FP32, [2, 100]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 25]))
        res = core.assign_group_by_size(var_list, [400])
        self.assertEqual([[0], [1], [2], [3]], res)

    def test_construct_group1(self):
        # multi dtype & one limit capability
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        res = core.assign_group_by_size(var_list, [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], res)

    def test_construct_group2(self):
        # one dtype & multi limit capability
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [2, 50]))
        res = core.assign_group_by_size(var_list, [400, 800])
        self.assertEqual([[0], [1, 2], [3]], res)

    def test_construct_group3(self):
        # multi dtype & multi limit capability
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        res = core.assign_group_by_size(var_list, [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5]], res)

    def test_construct_group4(self):
        # multi dtype & zero limit capability
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        res = core.assign_group_by_size(var_list, [0])
        self.assertEqual([[0], [1], [2], [3], [4], [5]], res)

    def test_construct_group5(self):
        # multi dtype & infinite capability
        var_list = []
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP32, [1, 50]))
        var_list.append(self.create_varbase(core.VarDesc.VarType.FP64, [1, 25]))
        res = core.assign_group_by_size(var_list, [10000])
        self.assertEqual([[0, 2, 4], [1, 3, 5]], res)


class TestDataParallelStateDict(unittest.TestCase):
    def test_data_parallel_state_dict(self):
        with fluid.dygraph.guard():
            strategy = paddle.distributed.prepare_context()
            mlp = MLP()
            parallel_mlp = dygraph.parallel.DataParallel(mlp, strategy)
            sgd = SGDOptimizer(
                learning_rate=1e-3, parameter_list=parallel_mlp.parameters())
            epoch_num = 2
            for epoch in range(epoch_num):
                data_numpy = np.random.random([1, 784])
                lablel_numpy = np.random.randint(1, 5, [10, 1])
                data_numpy = data_numpy.astype("float32")
                lablel_numpy = lablel_numpy.astype("float32")
                img = paddle.to_tensor(data_numpy)
                label = paddle.to_tensor(lablel_numpy)

                out = parallel_mlp(img)
                mse_loss = paddle.nn.loss.MSELoss()
                loss = mse_loss(input=out, label=label)
                loss.backward()
                sgd.minimize(loss)
                parallel_mlp.clear_gradients()


if __name__ == '__main__':
    unittest.main()
