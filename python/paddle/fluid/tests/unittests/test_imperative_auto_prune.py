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

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.nn import Embedding
from paddle.tensor import random


class AutoPruneLayer0(fluid.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            input_size,
            5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=2)
            ),
            bias_attr=False,
        )
        self.linear2 = paddle.nn.Linear(
            5,
            5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=2)
            ),
            bias_attr=False,
        )

    def forward(self, x, y):
        a = self.linear1(x)
        b = self.linear2(y)
        c = fluid.layers.mul(a, b)
        d = paddle.mean(c)
        return d


class AutoPruneLayer1(fluid.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            input_size,
            5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=2)
            ),
            bias_attr=False,
        )
        self.linear2 = paddle.nn.Linear(
            5,
            5,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=2)
            ),
            bias_attr=False,
        )

    def forward(self, x, y):
        a = self.linear1(x)
        b = self.linear2(y)
        b.stop_gradient = True
        c = fluid.layers.mul(a, b)
        d = paddle.mean(c)
        return d


class AutoPruneLayer2(fluid.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, 10)
        self.linear2 = paddle.nn.Linear(1, 1)

    def forward(self, x, label):
        feature = self.linear(x)
        label = self.linear2(label)
        label = fluid.layers.cast(label, dtype="float32")
        label = fluid.layers.cast(label, dtype='int64')
        # Note that the label is not persistable in paddle.nn.functional.cross_entropy.
        loss = paddle.nn.functional.cross_entropy(
            input=feature, label=label, reduction='none', use_softmax=False
        )
        loss = paddle.mean(loss)
        return loss


class AutoPruneLayer3(fluid.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.linear = paddle.nn.Linear(input_size, 20)

    def forward(self, x, label, test_num):
        feature = self.linear(x)
        part1, part2 = paddle.split(feature, num_or_sections=[10, 10], axis=1)
        # Note that: part2 is not used.
        loss = paddle.nn.functional.cross_entropy(
            input=part1, label=label, reduction='none', use_softmax=False
        )
        loss = paddle.mean(loss)
        if test_num == 1:
            return loss, part2
        else:
            return loss, part1, part2


class MyLayer(fluid.Layer):
    def __init__(self, input_size, vocab_size, size, dtype="float32"):
        super().__init__(dtype=dtype)
        self.embed0 = Embedding(vocab_size, size)
        self.embed1 = Embedding(vocab_size, size)
        self.linear_0 = paddle.nn.Linear(input_size, size)
        self.linear_1 = paddle.nn.Linear(input_size, size)

    def forward(self, x):
        # this method involves only the linear layers
        loss = paddle.mean(self.linear_0(x) + self.linear_1(x))
        return loss

    def linear0(self, x):
        loss = paddle.mean(self.linear_0(x))
        return loss

    def embed_linear0(self, x):
        loss = paddle.mean(self.linear_0(self.embed0(x)))
        return loss


class MyLayer2(fluid.Layer):
    def __init__(self, input_size, vocab_size, size, dtype="float32"):
        super().__init__(dtype=dtype)
        self.embed0 = Embedding(vocab_size, size)
        self.embed1 = Embedding(vocab_size, size)
        self.linear_0 = paddle.nn.Linear(input_size, size)
        self.linear_1 = paddle.nn.Linear(input_size, size)

    def forward(self, indices):
        # mind the difference with MyLayer
        # In this example, the forward method involes all params
        loss = paddle.mean(
            self.linear_0(self.embed0(indices))
            + self.linear_1(self.embed1(indices))
        )
        return loss

    def linear0(self, x):
        loss = paddle.mean(self.linear_0(x))
        return loss

    def embed_linear0(self, x):
        loss = paddle.mean(self.linear_0(self.embed0(x)))
        return loss


class TestImperativeAutoPrune(unittest.TestCase):
    def test_auto_prune(self):
        with fluid.dygraph.guard():
            case1 = AutoPruneLayer0(input_size=5)
            value1 = np.arange(25).reshape(5, 5).astype("float32")
            value2 = np.arange(25).reshape(5, 5).astype("float32")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss = case1(v1, v2)
            loss.backward()
            self.assertIsNotNone(case1.linear2.weight._grad_ivar())
            self.assertIsNotNone(case1.linear1.weight._grad_ivar())

    def test_auto_prune2(self):
        with fluid.dygraph.guard():
            case2 = AutoPruneLayer1(input_size=5)
            value1 = np.arange(25).reshape(5, 5).astype("float32")
            value2 = np.arange(25).reshape(5, 5).astype("float32")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss = case2(v1, v2)

            loss.backward()
            self.assertIsNone(case2.linear2.weight._grad_ivar())
            self.assertIsNotNone(case2.linear1.weight._grad_ivar())

    # TODO(jiabin): Support this when we support better split tensor
    def test_auto_prune3(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        with fluid.dygraph.guard():
            case3 = AutoPruneLayer3(input_size=784)
            value1 = np.arange(784).reshape(1, 784).astype("float32")
            value2 = np.arange(1).reshape(1, 1).astype("int64")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss, part2 = case3(v1, v2, 1)
            loss.backward()
            self.assertIsNotNone(case3.linear.weight._grad_ivar())
            self.assertTrue((part2.gradient() == 0).all())
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_auto_prune4(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        with fluid.dygraph.guard():
            case4 = AutoPruneLayer3(input_size=784)
            value1 = np.arange(784).reshape(1, 784).astype("float32")
            value2 = np.arange(1).reshape(1, 1).astype("int64")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss, part2 = case4(v1, v2, 1)
            part2.backward()
            self.assertIsNotNone(case4.linear.weight._grad_ivar())
            self.assertTrue((part2.gradient() == 1).all())
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_auto_prune5(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        with fluid.dygraph.guard():
            case4 = AutoPruneLayer3(input_size=784)
            value1 = np.arange(784).reshape(1, 784).astype("float32")
            value2 = np.arange(1).reshape(1, 1).astype("int64")
            v1 = fluid.dygraph.to_variable(value1)
            v2 = fluid.dygraph.to_variable(value2)
            loss, part1, part2 = case4(v1, v2, 2)
            part1.backward()
            self.assertIsNotNone(case4.linear.weight._grad_ivar())
            self.assertTrue((part2.gradient() == 0).all())
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_auto_prune6(self):
        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(3, 3)
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = linear(a)
            out2 = linear2(b)
            out1.stop_gradient = True
            out = fluid.layers.concat(input=[out1, out2, c], axis=1)
            out.backward()
            self.assertIsNone(linear.weight.gradient())
            self.assertIsNone(out1.gradient())

    def test_auto_prune7(self):
        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(3, 3)
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = linear(a)
            out2 = linear2(b)
            out1.stop_gradient = True
            out = fluid.layers.concat(input=[out1, out2, c], axis=1)
            out.backward()
            self.assertIsNone(linear.weight.gradient())
            self.assertIsNone(out1.gradient())

    def test_auto_prune8(self):
        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(5, 3)
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = linear(a)
            linear_origin = linear.weight.numpy()
            out2 = linear2(out1)
            linear2_origin = linear2.weight.numpy()
            linear2.weight.stop_gradient = True
            out2.backward()
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.003,
                parameter_list=(linear.parameters() + linear2.parameters()),
            )
            optimizer.minimize(out2)
            np.testing.assert_array_equal(
                linear2_origin, linear2.weight.numpy()
            )
            self.assertFalse(
                np.array_equal(linear_origin, linear.weight.numpy())
            )

    def test_auto_prune9(self):
        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(5, 3)
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = linear(a)
            linear_origin = linear.weight.numpy()
            out2 = linear2(out1)
            linear2_origin = linear2.weight.numpy()
            out2.stop_gradient = True
            out2.backward()
            optimizer = fluid.optimizer.SGD(
                learning_rate=0.003,
                parameter_list=(linear.parameters() + linear2.parameters()),
            )
            optimizer.minimize(out2)
            np.testing.assert_array_equal(
                linear2_origin, linear2.weight.numpy()
            )
            np.testing.assert_array_equal(linear_origin, linear.weight.numpy())
            try:
                linear2.weight.gradient()
            except ValueError as e:
                assert type(e) == ValueError

    def test_auto_prune10(self):
        with fluid.dygraph.guard():
            value0 = np.arange(26).reshape(2, 13).astype("float32")
            value1 = np.arange(6).reshape(2, 3).astype("float32")
            value2 = np.arange(10).reshape(2, 5).astype("float32")
            linear = paddle.nn.Linear(13, 5)
            linear2 = paddle.nn.Linear(3, 3)
            a = fluid.dygraph.to_variable(value0)
            b = fluid.dygraph.to_variable(value1)
            c = fluid.dygraph.to_variable(value2)
            out1 = linear(a)
            out2 = linear2(b)
            out1.stop_gradient = True
            out = fluid.layers.concat(input=[out1, out2, c], axis=1)
            # TODO(jiabin): In Eager Mode we don't actually need sort_sum_gradient, this test should be removed when we don't support fluid anymore.
            fluid.set_flags({'FLAGS_sort_sum_gradient': True})
            out.backward()
            self.assertIsNone(linear.weight.gradient())
            self.assertIsNone(out1.gradient())

    def test_auto_prune_with_optimizer(self):
        vocab_size = 100
        size = 20
        batch_size = 16

        indices = np.random.randint(
            low=0, high=100, size=(batch_size, 1)
        ).astype("int64")
        embed = np.random.randn(batch_size, size).astype("float32")

        place = fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            model = MyLayer(size, vocab_size, size)
            grad_clip = fluid.clip.GradientClipByGlobalNorm(0.001)
            optimizer = fluid.optimizer.AdamOptimizer(
                0.001, parameter_list=model.parameters(), grad_clip=grad_clip
            )
            indices = fluid.dygraph.to_variable(indices)
            embed = fluid.dygraph.to_variable(embed)
            dummy_loss = model(embed)

            loss = model.embed_linear0(indices)
            loss.backward()
            _, params_grads = optimizer.minimize(loss)
            for items in params_grads:
                assert items[0].name is not model.embed1.weight.name
                assert items[0].name is not model.linear_1.weight.name
            assert model.embed1.weight._grad_ivar() is None
            assert model.linear_1.weight._grad_ivar() is None

        with fluid.dygraph.guard(place):
            model = MyLayer2(size, vocab_size, size)
            grad_clip = fluid.clip.GradientClipByGlobalNorm(0.001)
            optimizer = fluid.optimizer.AdamOptimizer(
                0.001, parameter_list=model.parameters(), grad_clip=grad_clip
            )

            indices = fluid.dygraph.to_variable(indices)
            emebd = fluid.dygraph.to_variable(embed)
            dummy_loss = model(indices)

            loss = model.embed_linear0(indices)
            loss.backward()
            optimizer.minimize(loss)
            for items in params_grads:
                assert items[0].name is not model.embed1.weight.name
                assert items[0].name is not model.linear_1.weight.name
            assert model.embed1.weight._grad_ivar() is None
            assert model.linear_1.weight._grad_ivar() is None

    def test_case2_prune_no_grad_branch(self):
        with fluid.dygraph.guard():
            value1 = np.arange(784).reshape(1, 784)
            value2 = np.arange(1).reshape(1, 1)
            v1 = fluid.dygraph.to_variable(value1).astype("float32")
            v2 = fluid.dygraph.to_variable(value2).astype("float32")
            case3 = AutoPruneLayer2(input_size=784)
            loss = case3(v1, v2)
            loss.backward()
            self.assertIsNone(case3.linear2.weight._grad_ivar())
            self.assertIsNotNone(case3.linear.weight._grad_ivar())

    def test_case3_prune_no_grad_branch2(self):
        with fluid.dygraph.guard():
            value1 = np.arange(1).reshape(1, 1)
            linear = paddle.nn.Linear(1, 1)
            label = fluid.dygraph.to_variable(value1).astype("float32")
            label = linear(label)
            label = fluid.layers.cast(label, dtype="float32")
            label = fluid.layers.cast(label, dtype='int64')
            out = fluid.layers.one_hot(input=label, depth=100)
            loss = paddle.mean(out)
            loss.backward()
            self.assertIsNone(linear.weight._grad_ivar())

    def test_case4_with_no_grad_op_maker(self):
        with fluid.dygraph.guard():
            out = random.gaussian(shape=[20, 30])
            loss = paddle.mean(out)
            loss.backward()
            self.assertIsNone(out._grad_ivar())


if __name__ == '__main__':
    unittest.main()
