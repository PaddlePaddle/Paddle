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

import unittest
import paddle.fluid as fluid
import numpy as np


class AutoPruneLayer0(fluid.Layer):
    def __init__(self, name_scope):
        super(AutoPruneLayer0, self).__init__(name_scope)
        self.fc1 = fluid.dygraph.FC(
            "FC_1",
            5,
            param_attr=fluid.initializer.ConstantInitializer(value=2),
            bias_attr=False)
        self.fc2 = fluid.dygraph.FC(
            "FC_2",
            5,
            param_attr=fluid.initializer.ConstantInitializer(value=2),
            bias_attr=False)

    def forward(self, x, y):
        a = self.fc1(x)
        b = self.fc2(y)
        c = fluid.layers.mul(a, b)
        d = fluid.layers.reduce_mean(c)
        return d


class AutoPruneLayer1(fluid.Layer):
    def __init__(self, name_scope):
        super(AutoPruneLayer1, self).__init__(name_scope)
        self.fc1 = fluid.dygraph.FC(
            "FC_1",
            5,
            param_attr=fluid.initializer.ConstantInitializer(value=2),
            bias_attr=False)
        self.fc2 = fluid.dygraph.FC(
            "FC_2",
            5,
            param_attr=fluid.initializer.ConstantInitializer(value=2),
            bias_attr=False)

    def forward(self, x, y):
        a = self.fc1(x)
        b = self.fc2(y)
        b.stop_gradient = True
        c = fluid.layers.mul(a, b)
        d = fluid.layers.reduce_mean(c)
        return d


class AutoPruneLayer2(fluid.Layer):
    def __init__(self, name_scope):
        super(AutoPruneLayer2, self).__init__(name_scope)
        self.fc = fluid.layers.FC("FC1", size=10, act=None)

    def forward(self, x, label):
        feature = self.fc(x)
        label = fluid.layers.cast(label, dtype="float32")
        label = fluid.layers.cast(label, dtype='int64')
        # Note that the label is not persistable in fluid.layers.cross_entropy.
        loss = fluid.layers.cross_entropy(input=feature, label=label)
        loss = fluid.layers.mean(loss)
        return loss


class TestImperativeAutoPrune(unittest.TestCase):
    def test_auto_prune(self):
        with fluid.dygraph.guard():
            case1 = AutoPruneLayer1("l1")
            value1 = np.arange(25).reshape(5, 5).astype("float32")
            value2 = np.arange(25).reshape(5, 5).astype("float32")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss = case1(v1, v2)
            loss.backward()

    # def test_auto_prune2(self):
    #     with fluid.dygraph.guard():
    #         case1 = AutoPruneLayer1("l1")
    #         value1 = np.arange(25).reshape(5, 5).astype("float32")
    #         value2 = np.arange(25).reshape(5, 5).astype("float32")
    #         v1 = fluid.dygraph.to_variable(value1)
    #         v2 = fluid.dygraph.to_variable(value2)
    #         loss = case1(v1, v2)
    #         loss.backward()
    #
    # def case2_prune_no_grad_branch(self):
    #     with fluid.dygraph.guard():
    #         value1 = np.arange(784).reshape(1, 784)
    #         value2 = np.arange(10).reshape(1, 10)
    #         v1 = fluid.dygraph.to_variable(value1).astype("float32")
    #         v2 = fluid.dygraph.to_variable(value2).astype("float32")
    #         case2 = AutoPruneLayer2("l2")
    #         loss = case2(v1, v2)
    #         loss.backward()
    #
    # def case3_prune_no_grad_branch2(self):
    #     with fluid.dygraph.guard():
    #         value1 = np.arange(10).reshape(10, 1)
    #         label = fluid.dygraph.to_variable(value1).astype("float32")
    #         label = fluid.layers.cast(label, dtype="float32")
    #         label = fluid.layers.cast(label, dtype='int64')
    #         out = fluid.layers.one_hot(input=label, depth=100)
    #         loss = fluid.layers.mean(out)
    #         loss.backward()
    #
    # def case4_with_no_grad_op_maker(self):
    #     with fluid.dygraph.guard():
    #         out = fluid.layers.gaussian_random(shape=[20, 30])
    #         loss = fluid.layers.mean(out)
    #         loss.backward()


if __name__ == '__main__':
    unittest.main()
