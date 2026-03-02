# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict

import numpy as np
from op_test import get_device_place

import paddle
from paddle import base


def _make_param(shape, dtype='float32'):
    return paddle.create_parameter(shape=shape, dtype=dtype)


class MyLayer(paddle.nn.Layer):
    def __init__(self, num_stacked_param):
        super().__init__()
        # create ParameterDict with iterable Parameters
        self.params = self.paddle_imperative_ParameterDict(num_stacked_param)

    def paddle_imperative_ParameterDict(self, num_stacked_param):
        return paddle.nn.ParameterDict(
            [
                (
                    't' + str(i),
                    paddle.create_parameter(shape=[2, 2], dtype='float32'),
                )
                for i in range(num_stacked_param)
            ]
        )

    def forward(self, x):
        for i, key in enumerate(self.params):
            x = paddle.matmul(x, self.params[key])
        return x


class TestImperativeContainerParameterDict(unittest.TestCase):
    """Original test: basic forward/backward with list-of-tuples init and update."""

    def parameter_dict(self):
        self.place = get_device_place()
        data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
        with base.dygraph.guard():
            x = paddle.to_tensor(data_np)
            num_stacked_param = 4
            model = MyLayer(num_stacked_param)
            self.assertEqual(len(model.params), num_stacked_param)
            res = model(x)
            self.assertListEqual(res.shape, [5, 2])
            loss = paddle.mean(res)
            loss.backward()

            model.params['t' + str(num_stacked_param - 1)] = (
                paddle.create_parameter(shape=[2, 3], dtype='float32')
            )
            res = model(x)
            self.assertListEqual(res.shape, [5, 3])
            parameter = OrderedDict(
                [
                    (
                        't' + str(num_stacked_param),
                        paddle.create_parameter(shape=[3, 4], dtype='float32'),
                    )
                ]
            )
            model.params.update(parameter)
            self.assertEqual(len(model.params), num_stacked_param + 1)
            res = model(x)
            self.assertListEqual(res.shape, [5, 4])
            loss = paddle.mean(res)
            loss.backward()

    def test_parameter_dict(self):
        self.parameter_dict()


class TestParameterDictInit(unittest.TestCase):
    def test_init_types(self):
        # None, plain dict, OrderedDict, list of tuples
        self.assertEqual(len(paddle.nn.ParameterDict()), 0)
        self.assertEqual(
            len(paddle.nn.ParameterDict({'w': _make_param([2, 3])})), 1
        )
        self.assertEqual(
            len(
                paddle.nn.ParameterDict(
                    OrderedDict([('w', _make_param([2, 3]))])
                )
            ),
            1,
        )
        self.assertEqual(
            len(paddle.nn.ParameterDict([('w', _make_param([2, 3]))])), 1
        )

    def test_init_with_parameter_dict(self):
        # ParameterDict as input — exercises the update() fix
        pd1 = paddle.nn.ParameterDict({'w': _make_param([2, 3])})
        pd2 = paddle.nn.ParameterDict(pd1)
        self.assertEqual(len(pd2), 1)

    def test_init_with_values_alias(self):
        # @param_one_alias: 'values' maps to 'parameters'
        pd = paddle.nn.ParameterDict(values={'w': _make_param([2, 3])})
        self.assertEqual(len(pd), 1)

    def test_init_preserves_order(self):
        keys_in = ['c', 'a', 'b']
        pd = paddle.nn.ParameterDict(
            OrderedDict([(k, _make_param([1, 2])) for k in keys_in])
        )
        self.assertEqual(list(pd), keys_in)

    def test_init_errors(self):
        with self.assertRaises((ValueError, TypeError)):
            paddle.nn.ParameterDict([('w', _make_param([2, 3]), 'extra')])
        with self.assertRaises((AssertionError, TypeError)):
            paddle.nn.ParameterDict(42)


class TestParameterDictAccess(unittest.TestCase):
    def setUp(self):
        self.pd = paddle.nn.ParameterDict(
            {'w1': _make_param([2, 3]), 'w2': _make_param([3, 4])}
        )

    def test_getitem(self):
        self.assertEqual(list(self.pd['w1'].shape), [2, 3])
        self.assertEqual(list(self.pd['w2'].shape), [3, 4])

    def test_setitem(self):
        self.pd['w1'] = _make_param([2, 5])  # replace
        self.assertEqual(list(self.pd['w1'].shape), [2, 5])
        self.pd['w3'] = _make_param([4, 5])  # add new
        self.assertEqual(len(self.pd), 3)

    def test_setitem_non_parameter_raises(self):
        with self.assertRaises((AssertionError, TypeError)):
            self.pd['bad'] = paddle.to_tensor([1.0, 2.0])

    def test_len_iter_contains(self):
        self.assertEqual(len(self.pd), 2)
        self.assertEqual(sorted(self.pd), ['w1', 'w2'])
        self.assertIn('w1', self.pd)
        self.assertNotIn('missing', self.pd)


class TestParameterDictUpdate(unittest.TestCase):
    def setUp(self):
        self.pd = paddle.nn.ParameterDict({'w1': _make_param([2, 3])})

    def test_update_input_types(self):
        # plain dict, OrderedDict, list of tuples
        self.pd.update({'w2': _make_param([3, 4])})
        self.assertEqual(len(self.pd), 2)
        self.pd.update(OrderedDict([('w3', _make_param([4, 5]))]))
        self.assertEqual(len(self.pd), 3)
        self.pd.update([('w4', _make_param([5, 6]))])
        self.assertEqual(len(self.pd), 4)

    def test_update_from_parameter_dict(self):
        # ParameterDict as input — exercises the update() fix
        other = paddle.nn.ParameterDict({'w2': _make_param([3, 4])})
        self.pd.update(other)
        self.assertEqual(len(self.pd), 2)

    def test_update_overwrites(self):
        self.pd.update({'w1': _make_param([2, 5])})
        self.assertEqual(list(self.pd['w1'].shape), [2, 5])

    def test_update_errors(self):
        with self.assertRaises((ValueError, TypeError)):
            self.pd.update([('w2', _make_param([3, 4]), 'extra')])
        with self.assertRaises((AssertionError, TypeError)):
            self.pd.update(42)


class TestParameterDictRegistration(unittest.TestCase):
    def _make_model(self):
        class M(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.pd = paddle.nn.ParameterDict(
                    {'w1': _make_param([2, 3]), 'w2': _make_param([3, 4])}
                )

            def forward(self, x):
                return paddle.matmul(
                    paddle.matmul(x, self.pd['w1']), self.pd['w2']
                )

        return M()

    def test_registered_in_parameters_named_state_dict(self):
        model = self._make_model()
        self.assertEqual(len(list(model.parameters())), 2)
        named = dict(model.named_parameters())
        self.assertIn('pd.w1', named)
        self.assertIn('pd.w2', named)
        state = model.state_dict()
        self.assertIn('pd.w1', state)
        self.assertIn('pd.w2', state)

    def test_gradient_flows(self):
        model = self._make_model()
        paddle.matmul(paddle.uniform([2, 2]), model.pd['w1']).sum().backward()
        self.assertIsNotNone(model.pd['w1'].grad)

    def test_dynamic_setitem_and_update_registered(self):
        class M(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.pd = paddle.nn.ParameterDict()

            def forward(self, x):
                return x

        model = M()
        model.pd['w'] = _make_param([2, 2])
        model.pd.update({'v': _make_param([2, 2])})
        self.assertEqual(len(list(model.parameters())), 2)
        named = dict(model.named_parameters())
        self.assertIn('pd.w', named)
        self.assertIn('pd.v', named)


class TestParameterDictForwardBackward(unittest.TestCase):
    def _chain_model(self, n):
        class M(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.pd = paddle.nn.ParameterDict(
                    {f't{i}': _make_param([2, 2]) for i in range(n)}
                )

            def forward(self, x):
                for key in self.pd:
                    x = paddle.matmul(x, self.pd[key])
                return x

        return M()

    def test_forward_and_backward(self):
        model = self._chain_model(3)
        x = paddle.uniform([5, 2])
        out = model(x)
        self.assertEqual(list(out.shape), [5, 2])
        paddle.mean(out).backward()
        for key in model.pd:
            self.assertIsNotNone(model.pd[key].grad)

    def test_replace_param_changes_output_shape(self):
        model = self._chain_model(2)
        x = paddle.uniform([3, 2])
        self.assertEqual(list(model(x).shape), [3, 2])
        model.pd['t1'] = _make_param([2, 5])
        self.assertEqual(list(model(x).shape), [3, 5])

    def test_float64_params(self):
        pd = paddle.nn.ParameterDict(
            {'w': paddle.create_parameter(shape=[2, 3], dtype='float64')}
        )
        out = paddle.matmul(paddle.uniform([1, 2], dtype='float64'), pd['w'])
        self.assertEqual(list(out.shape), [1, 3])
        self.assertEqual(out.dtype, paddle.float64)


if __name__ == '__main__':
    unittest.main()
