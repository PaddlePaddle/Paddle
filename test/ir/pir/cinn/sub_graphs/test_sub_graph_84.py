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

# repo: PaddleDetection
# model: configs^picodet^legacy_model^picodet_s_320_coco_single_dy2st_train
# api||paddle.tensor.manipulation.split,api||paddle.tensor.manipulation.split,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,method||__sub__,method||clip,method||__sub__,method||clip,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__add__,method||__sub__,method||__add__,method||__truediv__,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,method||__sub__,method||__sub__,method||__mul__,method||__add__,method||__sub__,method||__truediv__,method||__sub__,method||__rsub__,method||__mul__
import unittest

import numpy as np

import paddle


class SIR285(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1596,  # (shape: [1, 4], dtype: paddle.float32, stop_gradient: False)
        var_1597,  # (shape: [1, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(
            var_1596, num_or_sections=4, axis=-1
        )
        var_1598 = out[0]
        var_1599 = out[1]
        var_1600 = out[2]
        var_1601 = out[3]
        out = paddle.tensor.manipulation.split(
            var_1597, num_or_sections=4, axis=-1
        )
        var_1602 = out[0]
        var_1603 = out[1]
        var_1604 = out[2]
        var_1605 = out[3]
        var_1606 = paddle.tensor.math.maximum(var_1598, var_1602)
        var_1607 = paddle.tensor.math.maximum(var_1599, var_1603)
        var_1608 = paddle.tensor.math.minimum(var_1600, var_1604)
        var_1609 = paddle.tensor.math.minimum(var_1601, var_1605)
        var_1610 = var_1608.__sub__(var_1606)
        var_1611 = var_1610.clip(0)
        var_1612 = var_1609.__sub__(var_1607)
        var_1613 = var_1612.clip(0)
        var_1614 = var_1611.__mul__(var_1613)
        var_1615 = var_1600.__sub__(var_1598)
        var_1616 = var_1601.__sub__(var_1599)
        var_1617 = var_1615.__mul__(var_1616)
        var_1618 = var_1604.__sub__(var_1602)
        var_1619 = var_1605.__sub__(var_1603)
        var_1620 = var_1618.__mul__(var_1619)
        var_1621 = var_1617.__add__(var_1620)
        var_1622 = var_1621.__sub__(var_1614)
        var_1623 = var_1622.__add__(1e-10)
        var_1624 = var_1614.__truediv__(var_1623)
        var_1625 = paddle.tensor.math.minimum(var_1598, var_1602)
        var_1626 = paddle.tensor.math.minimum(var_1599, var_1603)
        var_1627 = paddle.tensor.math.maximum(var_1600, var_1604)
        var_1628 = paddle.tensor.math.maximum(var_1601, var_1605)
        var_1629 = var_1627.__sub__(var_1625)
        var_1630 = var_1628.__sub__(var_1626)
        var_1631 = var_1629.__mul__(var_1630)
        var_1632 = var_1631.__add__(1e-10)
        var_1633 = var_1632.__sub__(var_1623)
        var_1634 = var_1633.__truediv__(var_1632)
        var_1635 = var_1624.__sub__(var_1634)
        var_1636 = var_1635.__rsub__(1)
        var_1637 = var_1636.__mul__(2.0)
        return var_1637


class TestSIR285(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 4], dtype=paddle.float32),
        )
        self.net = SIR285()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
