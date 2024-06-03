# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn


class MyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 300)

    def forward(self, x):
        return self.linear(x)

    @paddle.no_grad()
    def state_dict(
        self,
        destination=None,
        include_sublayers=True,
        structured_name_prefix="",
        use_hook=True,
        keep_vars=True,
    ):
        st = super().state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
            use_hook=use_hook,
            keep_vars=keep_vars,
        )
        st["linear.new_weight"] = paddle.transpose(
            st.pop("linear.weight"), [1, 0]
        )
        return st

    @paddle.no_grad()
    def set_state_dict(self, state_dict, use_structured_name=True):
        state_dict["linear.weight"] = paddle.transpose(
            state_dict.pop("linear.new_weight"), [1, 0]
        )
        return super().set_state_dict(state_dict)


class MyModel2(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 300)

    def forward(self, x):
        return self.linear(x)


def is_state_dict_equal(model1, model2):
    st1 = model1.state_dict()
    st2 = model2.state_dict()
    assert set(st1.keys()) == set(st2.keys())
    for k, v1 in st1.items():
        v2 = st2[k]
        if not np.array_equal(v1.numpy(), v2.numpy()):
            return False
    return True


class MyModel3(nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 300)
        buffer = paddle.to_tensor([0.0])
        self.register_buffer("model_buffer", buffer, persistable=True)

    def forward(self, x):
        return self.linear(x)


class TestStateDictConvert(unittest.TestCase):
    def test_main(self):
        model1 = MyModel()
        model2 = MyModel()
        self.assertFalse(is_state_dict_equal(model1, model2))
        model2.set_state_dict(model1.state_dict())
        self.assertTrue(is_state_dict_equal(model1, model2))


class TestStateDictReturn(unittest.TestCase):
    def test_missing_keys_and_unexpected_keys(self):
        model1 = MyModel2()
        tmp_dict = {}
        tmp_dict["unexpected_keys"] = paddle.to_tensor([1])
        missing_keys, unexpected_keys = model1.set_state_dict(tmp_dict)
        self.assertEqual(len(missing_keys), 2)
        self.assertEqual(missing_keys[0], "linear.weight")
        self.assertEqual(missing_keys[1], "linear.bias")
        self.assertEqual(len(unexpected_keys), 1)
        self.assertEqual(unexpected_keys[0], "unexpected_keys")


class TestStateKeepVars(unittest.TestCase):
    def test_true(self):
        model = MyModel3()
        x = paddle.randn([5, 100])
        y = model(x)
        y.backward()
        st = model.state_dict()
        has_grad = (
            (st["linear.weight"].grad == model.linear.weight.grad).all()
            and (st["linear.bias"].grad == model.linear.bias.grad).all()
            and st["model_buffer"].grad == model.model_buffer.grad
        )
        self.assertEqual(has_grad, True)

    def test_false(self):
        model = MyModel3()
        x = paddle.randn([5, 100])
        y = model(x)
        y.backward()
        st = model.state_dict(keep_vars=False)
        has_grad = (
            (st["linear.weight"].grad is not None)
            and (st["linear.bias"].grad is not None)
            and (st["model_buffer"].grad is not None)
        )
        self.assertEqual(has_grad, False)


if __name__ == "__main__":
    unittest.main()
