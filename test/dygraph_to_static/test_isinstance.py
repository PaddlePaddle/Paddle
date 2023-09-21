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

# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import unittest

import numpy as np
from dygraph_to_static_util import test_and_compare_with_new_ir

import paddle
from paddle import nn


class SimpleReturnLayer(nn.Layer):
    def forward(self, x):
        return x


class AddAttrLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.attr = None

    def forward(self, x):
        out = x + self.attr
        return out


class IsInstanceLayer(nn.Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    @paddle.jit.to_static
    def forward(self, x):
        if isinstance(self.layer, (AddAttrLayer,)):
            self.layer.attr = x
        res = self.layer(x)
        return res


class SequentialLayer(nn.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.LayerList(layers)

    @paddle.jit.to_static
    def forward(self, x):
        res = x
        for layer in self.layers:
            if isinstance(layer, AddAttrLayer):
                layer.attr = x
            res = layer(res)
        return res


@test_and_compare_with_new_ir(True)
def train(model, to_static):
    paddle.jit.enable_to_static(to_static)

    x = paddle.ones(shape=[2, 3], dtype='int32')
    out = model(x)

    return out.numpy()


class TestIsinstance(unittest.TestCase):
    def test_isinstance_simple_return_layer(self):
        model = IsInstanceLayer(SimpleReturnLayer())
        self._test_model(model)

    def test_isinstance_add_attr_layer(self):
        model = IsInstanceLayer(AddAttrLayer())
        self._test_model(model)

    def test_sequential_layer(self):
        layers = []
        for i in range(5):
            layers.append(SimpleReturnLayer())
            layers.append(AddAttrLayer())
        model = SequentialLayer(layers)
        self._test_model(model)

    def _test_model(self, model):
        st_out = train(model, to_static=True)
        dy_out = train(model, to_static=False)
        np.testing.assert_allclose(
            dy_out,
            st_out,
            rtol=1e-05,
            err_msg=f'dy_out:\n {dy_out}\n st_out:\n{st_out}',
        )


if __name__ == "__main__":
    unittest.main()
