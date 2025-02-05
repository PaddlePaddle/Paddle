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

import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
)

import paddle
from paddle import base

PLACE = base.CUDAPlace(0) if base.is_compiled_with_cuda() else base.CPUPlace()


class SubNetWithDict(paddle.nn.Layer):
    def __init__(self, hidden_size=16, output_size=16):
        super().__init__()

        init_weight = lambda x: paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(x)
        )

        self.q_fc = paddle.nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias_attr=False,
            weight_attr=init_weight(0.6),
        )
        self.k_fc = paddle.nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias_attr=False,
            weight_attr=init_weight(0.5),
        )
        self.v_fc = paddle.nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
            bias_attr=False,
            weight_attr=init_weight(0.2),
        )

    def forward(self, input, cache=None):
        input = paddle.to_tensor(input)

        q = self.q_fc(input)
        k = self.k_fc(input)
        v = self.v_fc(input)

        if cache is not None:
            cache_k, cache_v = cache["k"], cache["v"]
            k = 0.1 * cache_k + k
            v = 0.2 * cache_v + v
            cache["k"], cache["v"] = k, v

        weight = paddle.matmul(x=q, y=k, transpose_y=True)
        weight = paddle.nn.functional.softmax(weight)
        out = paddle.matmul(weight, v)

        return out


class MainNetWithDict(paddle.nn.Layer):
    def __init__(self, batch_size=64, hidden_size=16, output_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sub_net = SubNetWithDict(hidden_size, output_size)

    def forward(self, input, max_len=4):
        input = paddle.to_tensor(input)
        cache = {
            "k": paddle.tensor.fill_constant(
                shape=[self.batch_size, self.output_size],
                dtype='float32',
                value=0,
            ),
            "v": paddle.tensor.fill_constant(
                shape=[self.batch_size, self.output_size],
                dtype='float32',
                value=0,
            ),
        }
        # TODO(Aurelius84): The following code will be converted into:
        # max_len = paddle.static.nn.cond(paddle.shape(input)[0] != max_len,
        #                       lambda: paddle.shape(input)[0], lambda: max_len)
        # But max_len should be wrapped into tensor, which is not supported.

        # Comment out this line of code for now.
        # max_len = input.shape[0] if input.shape[0] != max_len else max_len
        out = input
        for i in range(max_len):
            out = self.sub_net(out, cache)
            cache = update_cache(cache)
        return out


# Test to call function defined outside of class.
def update_cache(cache):
    for k, val in cache.items():
        cache[k] = paddle.nn.functional.softmax(val)

    return cache


class TestNetWithDict(Dy2StTestBase):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.batch_size = self.x.shape[0]

    def _run_static(self):
        with enable_to_static_guard(True):
            return self.train()

    def _run_dygraph(self):
        with enable_to_static_guard(False):
            return self.train()

    def train(self):
        with base.dygraph.guard(PLACE):
            net = paddle.jit.to_static(
                MainNetWithDict(batch_size=self.batch_size)
            )
            ret = net(self.x)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


# Tests for dict pop
def test_dict_pop(x):
    x = paddle.to_tensor(x)
    dict_a = {"red": 0, "green": 1, "blue": 2}

    m = dict_a.pop("red")
    n = dict_a.pop("black", 3)

    out = x + m + n
    return out


def test_dict_pop_2(x):
    x = paddle.to_tensor(x)
    dict_a = {"red": x, "green": x + 1, "blue": x + 3}

    m = dict_a.pop("red")
    n = dict_a.pop("black", 3)

    out = x + m + n
    return out


class TestDictPop(Dy2StTestBase):
    def setUp(self):
        self.input = np.random.random(3).astype('int32')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self._set_test_func()

    def _set_test_func(self):
        self.dygraph_func = paddle.jit.to_static(test_dict_pop)

    def _run_static(self):
        return self._run(to_static=True)

    def _run_dygraph(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        with enable_to_static_guard(to_static):
            result = self.dygraph_func(self.input)
            return result.numpy()

    def test_transformed_result(self):
        dygraph_res = self._run_dygraph()
        static_res = self._run_static()
        np.testing.assert_allclose(
            dygraph_res,
            static_res,
            rtol=1e-05,
            err_msg=f'dygraph result is {dygraph_res}\nstatic result is {static_res}',
        )


class TestDictPop2(TestDictPop):
    def _set_test_func(self):
        self.dygraph_func = paddle.jit.to_static(test_dict_pop_2)


class NetWithDictPop(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        x = paddle.to_tensor(x)
        y = kwargs.pop('y', None)
        if y:
            y = paddle.to_tensor(x)
            x += y

        x.mean()
        return x


class TestDictPop3(TestNetWithDict):
    def setUp(self):
        self.x = np.array([2, 2]).astype('float32')

    def train(self):
        with base.dygraph.guard(PLACE):
            net = paddle.jit.to_static(NetWithDictPop())
            ret = net(z=0, x=self.x, y=True)
            return ret.numpy()

    def test_ast_to_func(self):
        dygraph_result = self._run_dygraph()
        static_result = self._run_static()

        self.assertTrue(
            (dygraph_result == static_result).all(),
            msg=f"dygraph result: {dygraph_result}\nstatic result: {static_result}",
        )


class TestDictCmpInFor(Dy2StTestBase):
    def test_with_for(self):
        def func():
            pos = [1, 3]
            neg = [-1, -3]
            dict_val = {'minus': 0}
            # test `zip` with `for`
            for x, y in zip(pos, neg):
                val = x - y
                dict_val.update(
                    {k: val + dict_val[k] for k, v in dict_val.items()}
                )

            return dict_val

        self.assertEqual(paddle.jit.to_static(func)()['minus'], 8)

    def test_with_for_enumerate(self):
        def func():
            pos = [1, 3]
            neg = [-1, -3]
            dict_val = {'minus': 0}
            # test `zip` with `for`
            for i, (x, y) in enumerate(zip(pos, neg)):
                val = x - y
                dict_val.update(
                    {k: val + dict_val[k] for k, v in dict_val.items()}
                )

            return dict_val

        self.assertEqual(paddle.jit.to_static(func)()['minus'], 8)


if __name__ == '__main__':
    unittest.main()
