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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^LeViT^LeViT_128
# api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,api||paddle.tensor.manipulation.concat,api||paddle.tensor.linalg.transpose,method||reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.matmul,method||__mul__,method||__add__,api||paddle.nn.functional.activation.softmax,api||paddle.tensor.linalg.matmul,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape
import unittest

import numpy as np

import paddle


class SIR93(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_1537 = self.create_parameter(
            shape=[8, 196],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_1532,  # (shape: [10, 49, 128], dtype: paddle.float32, stop_gradient: False)
        var_1533,  # (shape: [10, 8, 196, 16], dtype: paddle.float32, stop_gradient: False)
        var_1534,  # (shape: [10, 8, 196, 64], dtype: paddle.float32, stop_gradient: False)
        var_1538,  # (shape: [49, 196], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1535 = paddle.tensor.manipulation.reshape(var_1532, [10, 49, 8, 16])
        var_1536 = paddle.tensor.linalg.transpose(var_1535, perm=[0, 2, 1, 3])
        var_1539 = paddle.tensor.linalg.transpose(self.var_1537, (1, 0))
        var_1540 = var_1538.__getitem__(0)
        var_1541 = paddle.tensor.manipulation.gather(var_1539, var_1540)
        var_1542 = var_1538.__getitem__(1)
        var_1543 = paddle.tensor.manipulation.gather(var_1539, var_1542)
        var_1544 = var_1538.__getitem__(2)
        var_1545 = paddle.tensor.manipulation.gather(var_1539, var_1544)
        var_1546 = var_1538.__getitem__(3)
        var_1547 = paddle.tensor.manipulation.gather(var_1539, var_1546)
        var_1548 = var_1538.__getitem__(4)
        var_1549 = paddle.tensor.manipulation.gather(var_1539, var_1548)
        var_1550 = var_1538.__getitem__(5)
        var_1551 = paddle.tensor.manipulation.gather(var_1539, var_1550)
        var_1552 = var_1538.__getitem__(6)
        var_1553 = paddle.tensor.manipulation.gather(var_1539, var_1552)
        var_1554 = var_1538.__getitem__(7)
        var_1555 = paddle.tensor.manipulation.gather(var_1539, var_1554)
        var_1556 = var_1538.__getitem__(8)
        var_1557 = paddle.tensor.manipulation.gather(var_1539, var_1556)
        var_1558 = var_1538.__getitem__(9)
        var_1559 = paddle.tensor.manipulation.gather(var_1539, var_1558)
        var_1560 = var_1538.__getitem__(10)
        var_1561 = paddle.tensor.manipulation.gather(var_1539, var_1560)
        var_1562 = var_1538.__getitem__(11)
        var_1563 = paddle.tensor.manipulation.gather(var_1539, var_1562)
        var_1564 = var_1538.__getitem__(12)
        var_1565 = paddle.tensor.manipulation.gather(var_1539, var_1564)
        var_1566 = var_1538.__getitem__(13)
        var_1567 = paddle.tensor.manipulation.gather(var_1539, var_1566)
        var_1568 = var_1538.__getitem__(14)
        var_1569 = paddle.tensor.manipulation.gather(var_1539, var_1568)
        var_1570 = var_1538.__getitem__(15)
        var_1571 = paddle.tensor.manipulation.gather(var_1539, var_1570)
        var_1572 = var_1538.__getitem__(16)
        var_1573 = paddle.tensor.manipulation.gather(var_1539, var_1572)
        var_1574 = var_1538.__getitem__(17)
        var_1575 = paddle.tensor.manipulation.gather(var_1539, var_1574)
        var_1576 = var_1538.__getitem__(18)
        var_1577 = paddle.tensor.manipulation.gather(var_1539, var_1576)
        var_1578 = var_1538.__getitem__(19)
        var_1579 = paddle.tensor.manipulation.gather(var_1539, var_1578)
        var_1580 = var_1538.__getitem__(20)
        var_1581 = paddle.tensor.manipulation.gather(var_1539, var_1580)
        var_1582 = var_1538.__getitem__(21)
        var_1583 = paddle.tensor.manipulation.gather(var_1539, var_1582)
        var_1584 = var_1538.__getitem__(22)
        var_1585 = paddle.tensor.manipulation.gather(var_1539, var_1584)
        var_1586 = var_1538.__getitem__(23)
        var_1587 = paddle.tensor.manipulation.gather(var_1539, var_1586)
        var_1588 = var_1538.__getitem__(24)
        var_1589 = paddle.tensor.manipulation.gather(var_1539, var_1588)
        var_1590 = var_1538.__getitem__(25)
        var_1591 = paddle.tensor.manipulation.gather(var_1539, var_1590)
        var_1592 = var_1538.__getitem__(26)
        var_1593 = paddle.tensor.manipulation.gather(var_1539, var_1592)
        var_1594 = var_1538.__getitem__(27)
        var_1595 = paddle.tensor.manipulation.gather(var_1539, var_1594)
        var_1596 = var_1538.__getitem__(28)
        var_1597 = paddle.tensor.manipulation.gather(var_1539, var_1596)
        var_1598 = var_1538.__getitem__(29)
        var_1599 = paddle.tensor.manipulation.gather(var_1539, var_1598)
        var_1600 = var_1538.__getitem__(30)
        var_1601 = paddle.tensor.manipulation.gather(var_1539, var_1600)
        var_1602 = var_1538.__getitem__(31)
        var_1603 = paddle.tensor.manipulation.gather(var_1539, var_1602)
        var_1604 = var_1538.__getitem__(32)
        var_1605 = paddle.tensor.manipulation.gather(var_1539, var_1604)
        var_1606 = var_1538.__getitem__(33)
        var_1607 = paddle.tensor.manipulation.gather(var_1539, var_1606)
        var_1608 = var_1538.__getitem__(34)
        var_1609 = paddle.tensor.manipulation.gather(var_1539, var_1608)
        var_1610 = var_1538.__getitem__(35)
        var_1611 = paddle.tensor.manipulation.gather(var_1539, var_1610)
        var_1612 = var_1538.__getitem__(36)
        var_1613 = paddle.tensor.manipulation.gather(var_1539, var_1612)
        var_1614 = var_1538.__getitem__(37)
        var_1615 = paddle.tensor.manipulation.gather(var_1539, var_1614)
        var_1616 = var_1538.__getitem__(38)
        var_1617 = paddle.tensor.manipulation.gather(var_1539, var_1616)
        var_1618 = var_1538.__getitem__(39)
        var_1619 = paddle.tensor.manipulation.gather(var_1539, var_1618)
        var_1620 = var_1538.__getitem__(40)
        var_1621 = paddle.tensor.manipulation.gather(var_1539, var_1620)
        var_1622 = var_1538.__getitem__(41)
        var_1623 = paddle.tensor.manipulation.gather(var_1539, var_1622)
        var_1624 = var_1538.__getitem__(42)
        var_1625 = paddle.tensor.manipulation.gather(var_1539, var_1624)
        var_1626 = var_1538.__getitem__(43)
        var_1627 = paddle.tensor.manipulation.gather(var_1539, var_1626)
        var_1628 = var_1538.__getitem__(44)
        var_1629 = paddle.tensor.manipulation.gather(var_1539, var_1628)
        var_1630 = var_1538.__getitem__(45)
        var_1631 = paddle.tensor.manipulation.gather(var_1539, var_1630)
        var_1632 = var_1538.__getitem__(46)
        var_1633 = paddle.tensor.manipulation.gather(var_1539, var_1632)
        var_1634 = var_1538.__getitem__(47)
        var_1635 = paddle.tensor.manipulation.gather(var_1539, var_1634)
        var_1636 = var_1538.__getitem__(48)
        var_1637 = paddle.tensor.manipulation.gather(var_1539, var_1636)
        var_1638 = paddle.tensor.manipulation.concat(
            [
                var_1541,
                var_1543,
                var_1545,
                var_1547,
                var_1549,
                var_1551,
                var_1553,
                var_1555,
                var_1557,
                var_1559,
                var_1561,
                var_1563,
                var_1565,
                var_1567,
                var_1569,
                var_1571,
                var_1573,
                var_1575,
                var_1577,
                var_1579,
                var_1581,
                var_1583,
                var_1585,
                var_1587,
                var_1589,
                var_1591,
                var_1593,
                var_1595,
                var_1597,
                var_1599,
                var_1601,
                var_1603,
                var_1605,
                var_1607,
                var_1609,
                var_1611,
                var_1613,
                var_1615,
                var_1617,
                var_1619,
                var_1621,
                var_1623,
                var_1625,
                var_1627,
                var_1629,
                var_1631,
                var_1633,
                var_1635,
                var_1637,
            ]
        )
        var_1639 = paddle.tensor.linalg.transpose(var_1638, (1, 0))
        var_1640 = var_1639.reshape((0, 49, 196))
        var_1641 = paddle.tensor.linalg.transpose(var_1533, perm=[0, 1, 3, 2])
        var_1642 = paddle.tensor.linalg.matmul(var_1536, var_1641)
        var_1643 = var_1642.__mul__(0.25)
        var_1644 = var_1643.__add__(var_1640)
        var_1645 = paddle.nn.functional.activation.softmax(var_1644)
        var_1646 = paddle.tensor.linalg.matmul(var_1645, var_1534)
        var_1647 = paddle.tensor.linalg.transpose(var_1646, perm=[0, 2, 1, 3])
        var_1648 = paddle.tensor.manipulation.reshape(var_1647, [10, -1, 512])
        return var_1648


class TestSIR93(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[10, 49, 128], dtype=paddle.float32),
            paddle.rand(shape=[10, 8, 196, 16], dtype=paddle.float32),
            paddle.rand(shape=[10, 8, 196, 64], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[49, 196], dtype=paddle.int64),
        )
        self.net = SIR93()

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
