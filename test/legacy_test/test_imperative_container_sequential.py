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
from paddle import base
from paddle.nn import Linear


class TestImperativeContainerSequential(unittest.TestCase):
    def test_sequential(self):
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with base.dygraph.guard():
            data = paddle.to_tensor(data)
            model1 = paddle.nn.Sequential(Linear(10, 1), Linear(1, 2))
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 2])
            model1[1] = Linear(1, 3)
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 3])
            loss1 = paddle.mean(res1)
            loss1.backward()

            l1 = Linear(10, 1)
            l2 = Linear(1, 3)
            model2 = paddle.nn.Sequential(('l1', l1), ('l2', l2))
            self.assertEqual(len(model2), 2)
            res2 = model2(data)
            self.assertTrue(l1 is model2.l1)
            self.assertListEqual(res2.shape, res1.shape)
            self.assertEqual(len(model1.parameters()), len(model2.parameters()))
            del model2['l2']
            self.assertEqual(len(model2), 1)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 1])
            model2.add_sublayer('l3', Linear(1, 3))
            model2.add_sublayer('l4', Linear(3, 4))
            self.assertEqual(len(model2), 3)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 4])

            loss2 = paddle.mean(res2)
            loss2.backward()

    def test_append_insert_extend(self):
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with base.dygraph.guard():
            data = paddle.to_tensor(data)

            model1 = paddle.nn.Sequential()
            # test append
            model1.append(Linear(10, 1))
            model1.append(Linear(1, 2))
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 2])

            # test insert
            model1.insert(0, Linear(10, 10))
            res1 = model1(data)

            # test insert type error(non nn.Layer type)
            model2 = paddle.nn.Sequential()
            self.assertRaises(AssertionError, model2.insert, 0, 1)

            # test insert index error(1)
            model2 = paddle.nn.Sequential()
            self.assertRaises(IndexError, model2.insert, 1, Linear(10, 10))

            # test insert at negative index -1
            model2 = paddle.nn.Sequential()
            model2.insert(0, Linear(10, 10))
            self.assertEqual(len(model2), 1)

            # res1 = model1(data)

            # test extend
            model1.extend([Linear(2, 3), Linear(3, 4)])
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 4])

            loss1 = paddle.mean(res1)
            loss1.backward()

            # test __iter__
            model3 = paddle.nn.Sequential(
                Linear(10, 1),
                Linear(1, 2),
            )
            output1 = model3(data)
            output2 = data
            for layer in model3:
                output2 = layer(output2)
            np.testing.assert_allclose(
                output1.numpy(), output2.numpy(), equal_nan=True
            )

    def test_sequential_list_params(self):
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with base.dygraph.guard():
            data = paddle.to_tensor(data)
            model1 = paddle.nn.Sequential(Linear(10, 1), Linear(1, 2))
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 2])
            model1[1] = Linear(1, 3)
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 3])
            loss1 = paddle.mean(res1)
            loss1.backward()

            l1 = Linear(10, 1)
            l2 = Linear(1, 3)
            model2 = paddle.nn.Sequential(['l1', l1], ['l2', l2])
            self.assertEqual(len(model2), 2)
            res2 = model2(data)
            self.assertTrue(l1 is model2.l1)
            self.assertListEqual(res2.shape, res1.shape)
            self.assertEqual(len(model1.parameters()), len(model2.parameters()))
            del model2['l2']
            self.assertEqual(len(model2), 1)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 1])
            model2.add_sublayer('l3', Linear(1, 3))
            model2.add_sublayer('l4', Linear(3, 4))
            self.assertEqual(len(model2), 3)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 4])

            loss2 = paddle.mean(res2)
            loss2.backward()


if __name__ == '__main__':
    unittest.main()
