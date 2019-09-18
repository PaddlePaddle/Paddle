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
        self.fc = fluid.dygraph.FC("FC1", size=10, act=None)
        self.fc2 = fluid.dygraph.FC("FC2", size=1, act=None)

    def forward(self, x, label):
        feature = self.fc(x)
        label = self.fc2(label)
        label = fluid.layers.cast(label, dtype="float32")
        label = fluid.layers.cast(label, dtype='int64')
        # Note that the label is not persistable in fluid.layers.cross_entropy.
        loss = fluid.layers.cross_entropy(input=feature, label=label)
        loss = fluid.layers.mean(loss)
        return loss


class AutoPruneLayer3(fluid.Layer):
    def __init__(self, name_scope):
        super(AutoPruneLayer3, self).__init__(name_scope)
        self.fc = fluid.dygraph.FC("FC1", size=20, act=None)

    def forward(self, x, label, test_num):
        feature = self.fc(x)
        part1, part2 = fluid.layers.split(
            feature, num_or_sections=[10, 10], dim=1)
        # Note that: part2 is not used.
        loss = fluid.layers.cross_entropy(input=part1, label=label)
        loss = fluid.layers.mean(loss)
        if test_num == 1:
            return loss, part2
        else:
            return loss, part1, part2


class MyLayer(fluid.Layer):
    def __init__(self, name_scope, vocab_size, size, dtype="float32"):
        super(MyLayer, self).__init__(name_scope, dtype)
        self.embed0 = fluid.Embedding(self.full_name(), size=(vocab_size, size))
        self.embed1 = fluid.Embedding(self.full_name(), size=(vocab_size, size))
        self.fc0 = fluid.FC(self.full_name(), size=size, dtype=dtype)
        self.fc1 = fluid.FC(self.full_name(), size=size, dtype=dtype)

    def forward(self, x):
        # this method involves only the fc layers
        loss = fluid.layers.reduce_mean(self.fc0(x) + self.fc1(x))
        return loss

    def linear0(self, x):
        loss = fluid.layers.reduce_mean(self.fc0(x))
        return loss

    def embed_linear0(self, x):
        loss = fluid.layers.reduce_mean(self.fc0(self.embed0(x)))
        return loss


class MyLayer2(fluid.Layer):
    def __init__(self, name_scope, vocab_size, size, dtype="float32"):
        super(MyLayer2, self).__init__(name_scope, dtype)
        self.embed0 = fluid.Embedding(self.full_name(), size=(vocab_size, size))
        self.embed1 = fluid.Embedding(self.full_name(), size=(vocab_size, size))
        self.fc0 = fluid.FC(self.full_name(), size=size, dtype=dtype)
        self.fc1 = fluid.FC(self.full_name(), size=size, dtype=dtype)

    def forward(self, indices):
        # mind the difference with MyLayer
        # In this example, the forward method involes all params
        loss = fluid.layers.reduce_mean(
            self.fc0(self.embed0(indices)) + self.fc1(self.embed1(indices)))
        return loss

    def linear0(self, x):
        loss = fluid.layers.reduce_mean(self.fc0(x))
        return loss

    def embed_linear0(self, x):
        loss = fluid.layers.reduce_mean(self.fc0(self.embed0(x)))
        return loss


class TestImperativeAutoPrune(unittest.TestCase):
    def test_auto_prune(self):
        with fluid.dygraph.guard():
            case1 = AutoPruneLayer0("l1")
            value1 = np.arange(25).reshape(5, 5).astype("float32")
            value2 = np.arange(25).reshape(5, 5).astype("float32")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss = case1(v1, v2)
            loss.backward()
            self.assertTrue(case1.fc2._w._ivar._grad_ivar() is not None)
            self.assertTrue(case1.fc1._w._ivar._grad_ivar() is not None)

    def test_auto_prune2(self):
        with fluid.dygraph.guard():
            case2 = AutoPruneLayer1("l1")
            value1 = np.arange(25).reshape(5, 5).astype("float32")
            value2 = np.arange(25).reshape(5, 5).astype("float32")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss = case2(v1, v2)
            loss.backward()
            self.assertTrue(case2.fc2._w._ivar._grad_ivar() is None)
            self.assertTrue(case2.fc1._w._ivar._grad_ivar() is not None)

    def test_auto_prune3(self):
        with fluid.dygraph.guard():
            case3 = AutoPruneLayer3("l3")
            value1 = np.arange(784).reshape(1, 784).astype("float32")
            value2 = np.arange(1).reshape(1, 1).astype("int64")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss, part2 = case3(v1, v2, 1)
            loss.backward()
            self.assertTrue(case3.fc._w._ivar._grad_ivar() is not None)
            self.assertTrue((part2.gradient() == 0).all())

    def test_auto_prune4(self):
        with fluid.dygraph.guard():
            case4 = AutoPruneLayer3("l3")
            value1 = np.arange(784).reshape(1, 784).astype("float32")
            value2 = np.arange(1).reshape(1, 1).astype("int64")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss, part2 = case4(v1, v2, 1)
            part2.backward()
            self.assertTrue(case4.fc._w._ivar._grad_ivar() is not None)
            self.assertTrue((part2.gradient() == 1).all())

    def test_auto_prune5(self):
        with fluid.dygraph.guard():
            case4 = AutoPruneLayer3("l3")
            value1 = np.arange(784).reshape(1, 784).astype("float32")
            value2 = np.arange(1).reshape(1, 1).astype("int64")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss, part1, part2 = case4(v1, v2, 2)
            part1.backward()
            self.assertTrue(case4.fc._w._ivar._grad_ivar() is not None)
            self.assertTrue((part2.gradient() == 0).all())

    def test_auto_prune6(self):
        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            fc = fluid.FC("fc1", size=5, dtype="float32")
            fc2 = fluid.FC("fc2", size=3, dtype="float32")
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = fc(a)
            out2 = fc2(b)
            out1.stop_gradient = True
            out = fluid.layers.concat(input=[out1, out2, c], axis=1)
            out.backward()
            self.assertTrue((fc._w.gradient() == 0).all())
            self.assertTrue((out1.gradient() == 0).all())

    def test_auto_prune7(self):
        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            fc = fluid.FC("fc1", size=5, dtype="float32")
            fc2 = fluid.FC("fc2", size=3, dtype="float32")
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = fc(a)
            out2 = fc2(b)
            out1.stop_gradient = True
            out = fluid.layers.concat(input=[out1, out2, c], axis=1)
            backward_strategy = fluid.dygraph.BackwardStrategy()
            out.backward(backward_strategy)
            self.assertTrue((fc._w.gradient() == 0).all())
            self.assertTrue((out1.gradient() == 0).all())

    def test_auto_prune_with_optimizer(self):
        vocab_size = 100
        size = 20
        batch_size = 16

        indices = np.random.randint(
            low=0, high=100, size=(batch_size, 1)).astype("int64")
        embed = np.random.randn(batch_size, size).astype("float32")

        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            model = MyLayer("mylayer", vocab_size, size)
            optimizer = fluid.optimizer.AdamOptimizer(0.001)
            grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(0.001)

            indices = fluid.dygraph.to_variable(indices)
            emebd = fluid.dygraph.to_variable(embed)
            dummy_loss = model(embed)

            loss = model.embed_linear0(indices)
            loss.backward()
            _, params_grads = optimizer.minimize(loss, grad_clip=grad_clip)
            for items in params_grads:
                assert items[0].name is not model.embed1._w.name
                assert items[0].name is not model.fc1._w.name
            assert model.embed1._w._ivar._grad_ivar() is None
            assert model.fc1._w._ivar._grad_ivar() is None

        with fluid.dygraph.guard(place):
            model = MyLayer2("mylayer", vocab_size, size)
            optimizer = fluid.optimizer.AdamOptimizer(0.001)
            grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(0.001)

            indices = fluid.dygraph.to_variable(indices)
            emebd = fluid.dygraph.to_variable(embed)
            dummy_loss = model(indices)

            loss = model.embed_linear0(indices)
            loss.backward()
            optimizer.minimize(loss, grad_clip=grad_clip)
            for items in params_grads:
                assert items[0].name is not model.embed1._w.name
                assert items[0].name is not model.fc1._w.name
            assert model.embed1._w._ivar._grad_ivar() is None
            assert model.fc1._w._ivar._grad_ivar() is None

    def test_case2_prune_no_grad_branch(self):
        with fluid.dygraph.guard():
            value1 = np.arange(784).reshape(1, 784)
            value2 = np.arange(1).reshape(1, 1)
            v1 = fluid.dygraph.to_variable(value1).astype("float32")
            v2 = fluid.dygraph.to_variable(value2).astype("float32")
            case3 = AutoPruneLayer2("l2")
            loss = case3(v1, v2)
            loss.backward()
            self.assertTrue(case3.fc2._w._ivar._grad_ivar() is None)
            self.assertTrue(case3.fc._w._ivar._grad_ivar() is not None)

    def test_case2_prune_no_grad_branch(self):
        with fluid.dygraph.guard():
            value1 = np.arange(784).reshape(1, 784)
            value2 = np.arange(1).reshape(1, 1)
            v1 = fluid.dygraph.to_variable(value1).astype("float32")
            v2 = fluid.dygraph.to_variable(value2).astype("float32")
            case3 = AutoPruneLayer2("l2")
            loss = case3(v1, v2)
            loss.backward()
            self.assertTrue(case3.fc2._w._ivar._grad_ivar() is None)
            self.assertTrue(case3.fc._w._ivar._grad_ivar() is not None)

    def test_case3_prune_no_grad_branch2(self):
        with fluid.dygraph.guard():
            value1 = np.arange(1).reshape(1, 1)
            fc = fluid.dygraph.FC("FC1", size=1, act=None)
            label = fluid.dygraph.to_variable(value1).astype("float32")
            label = fc(label)
            label = fluid.layers.cast(label, dtype="float32")
            label = fluid.layers.cast(label, dtype='int64')
            out = fluid.layers.one_hot(input=label, depth=100)
            loss = fluid.layers.mean(out)
            loss.backward()
            self.assertTrue(fc._w._ivar._grad_ivar() is None)

    def test_case4_with_no_grad_op_maker(self):
        with fluid.dygraph.guard():
            out = fluid.layers.gaussian_random(shape=[20, 30])
            loss = fluid.layers.mean(out)
            loss.backward()
            self.assertTrue(out._ivar._grad_ivar() is None)


if __name__ == '__main__':
    unittest.main()
