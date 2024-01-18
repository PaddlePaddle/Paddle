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
# model: configs^yolox^yolox_l_300e_coco_single_dy2st_train
# api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.manipulation.concat,method||astype,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.split,method||__truediv__,method||__add__,api||paddle.tensor.ops.exp,method||__mul__,method||__sub__,method||__add__,api||paddle.tensor.manipulation.concat,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.manipulation.concat,method||astype,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR122(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_582,  # (shape: [1, 5184, 80], dtype: paddle.float32, stop_gradient: False)
        var_583,  # (shape: [1, 1296, 80], dtype: paddle.float32, stop_gradient: False)
        var_584,  # (shape: [1, 324, 80], dtype: paddle.float32, stop_gradient: False)
        var_585,  # (shape: [1, 5184, 4], dtype: paddle.float32, stop_gradient: False)
        var_586,  # (shape: [1, 1296, 4], dtype: paddle.float32, stop_gradient: False)
        var_587,  # (shape: [1, 324, 4], dtype: paddle.float32, stop_gradient: False)
        var_588,  # (shape: [1, 5184, 1], dtype: paddle.float32, stop_gradient: False)
        var_589,  # (shape: [1, 1296, 1], dtype: paddle.float32, stop_gradient: False)
        var_590,  # (shape: [1, 324, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_591 = paddle.tensor.manipulation.concat(
            [var_582, var_583, var_584], axis=1
        )
        var_592 = paddle.tensor.manipulation.concat(
            [var_585, var_586, var_587], axis=1
        )
        var_593 = paddle.tensor.manipulation.concat(
            [var_588, var_589, var_590], axis=1
        )
        var_594 = paddle.tensor.creation.arange(72)
        var_595 = var_594.__add__(0.0)
        var_596 = var_595.__mul__(8)
        var_597 = paddle.tensor.creation.arange(72)
        var_598 = var_597.__add__(0.0)
        var_599 = var_598.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_599, var_596)
        var_600 = out[0]
        var_601 = out[1]
        var_602 = paddle.tensor.manipulation.stack([var_601, var_600], axis=-1)
        var_603 = var_602.reshape([-1, 2])
        var_604 = paddle.tensor.creation.full([5184, 1], 8, dtype='float32')
        var_605 = paddle.tensor.creation.arange(36)
        var_606 = var_605.__add__(0.0)
        var_607 = var_606.__mul__(16)
        var_608 = paddle.tensor.creation.arange(36)
        var_609 = var_608.__add__(0.0)
        var_610 = var_609.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_610, var_607)
        var_611 = out[0]
        var_612 = out[1]
        var_613 = paddle.tensor.manipulation.stack([var_612, var_611], axis=-1)
        var_614 = var_613.reshape([-1, 2])
        var_615 = paddle.tensor.creation.full([1296, 1], 16, dtype='float32')
        var_616 = paddle.tensor.creation.arange(18)
        var_617 = var_616.__add__(0.0)
        var_618 = var_617.__mul__(32)
        var_619 = paddle.tensor.creation.arange(18)
        var_620 = var_619.__add__(0.0)
        var_621 = var_620.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_621, var_618)
        var_622 = out[0]
        var_623 = out[1]
        var_624 = paddle.tensor.manipulation.stack([var_623, var_622], axis=-1)
        var_625 = var_624.reshape([-1, 2])
        var_626 = paddle.tensor.creation.full([324, 1], 32, dtype='float32')
        var_627 = paddle.tensor.manipulation.concat([var_603, var_614, var_625])
        var_628 = var_627.astype('float32')
        var_629 = paddle.tensor.manipulation.concat([var_604, var_615, var_626])
        out = paddle.tensor.manipulation.split(var_592, 2, axis=-1)
        var_630 = out[0]
        var_631 = out[1]
        var_632 = var_628.__truediv__(var_629)
        var_633 = var_630.__add__(var_632)
        var_634 = paddle.tensor.ops.exp(var_631)
        var_635 = var_634.__mul__(0.5)
        var_636 = var_633.__sub__(var_635)
        var_637 = var_633.__add__(var_635)
        var_638 = paddle.tensor.manipulation.concat([var_636, var_637], axis=-1)
        var_639 = paddle.tensor.creation.arange(72)
        var_640 = var_639.__add__(0.5)
        var_641 = var_640.__mul__(8)
        var_642 = paddle.tensor.creation.arange(72)
        var_643 = var_642.__add__(0.5)
        var_644 = var_643.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_644, var_641)
        var_645 = out[0]
        var_646 = out[1]
        var_647 = paddle.tensor.manipulation.stack([var_646, var_645], axis=-1)
        var_648 = var_647.reshape([-1, 2])
        var_649 = paddle.tensor.creation.full([5184, 1], 8, dtype='float32')
        var_650 = paddle.tensor.creation.arange(36)
        var_651 = var_650.__add__(0.5)
        var_652 = var_651.__mul__(16)
        var_653 = paddle.tensor.creation.arange(36)
        var_654 = var_653.__add__(0.5)
        var_655 = var_654.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_655, var_652)
        var_656 = out[0]
        var_657 = out[1]
        var_658 = paddle.tensor.manipulation.stack([var_657, var_656], axis=-1)
        var_659 = var_658.reshape([-1, 2])
        var_660 = paddle.tensor.creation.full([1296, 1], 16, dtype='float32')
        var_661 = paddle.tensor.creation.arange(18)
        var_662 = var_661.__add__(0.5)
        var_663 = var_662.__mul__(32)
        var_664 = paddle.tensor.creation.arange(18)
        var_665 = var_664.__add__(0.5)
        var_666 = var_665.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_666, var_663)
        var_667 = out[0]
        var_668 = out[1]
        var_669 = paddle.tensor.manipulation.stack([var_668, var_667], axis=-1)
        var_670 = var_669.reshape([-1, 2])
        var_671 = paddle.tensor.creation.full([324, 1], 32, dtype='float32')
        var_672 = paddle.tensor.manipulation.concat([var_648, var_659, var_670])
        var_673 = var_672.astype('float32')
        var_674 = paddle.tensor.manipulation.concat([var_649, var_660, var_671])
        return (
            var_591,
            var_638,
            var_593,
            var_673,
            var_674,
            var_648,
            var_659,
            var_670,
        )


class TestSIR122(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 5184, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 1296, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 324, 80], dtype=paddle.float32),
            paddle.rand(shape=[1, 5184, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 1296, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 324, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 5184, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 1296, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 324, 1], dtype=paddle.float32),
        )
        self.net = SIR122()

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
