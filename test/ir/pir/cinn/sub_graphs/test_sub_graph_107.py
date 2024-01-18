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
# model: configs^ppyoloe^ppyoloe_crn_l_300e_coco_single_dy2st_train
# api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR157(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        var_653 = paddle.tensor.creation.arange(end=14)
        var_654 = var_653.__add__(0.5)
        var_655 = var_654.__mul__(32)
        var_656 = paddle.tensor.creation.arange(end=14)
        var_657 = var_656.__add__(0.5)
        var_658 = var_657.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_658, var_655)
        var_659 = out[0]
        var_660 = out[1]
        var_661 = var_660.__sub__(80.0)
        var_662 = var_659.__sub__(80.0)
        var_663 = var_660.__add__(80.0)
        var_664 = var_659.__add__(80.0)
        var_665 = paddle.tensor.manipulation.stack(
            [var_661, var_662, var_663, var_664], axis=-1
        )
        var_666 = var_665.astype('float32')
        var_667 = paddle.tensor.manipulation.stack([var_660, var_659], axis=-1)
        var_668 = var_667.astype('float32')
        var_669 = var_666.reshape([-1, 4])
        var_670 = var_668.reshape([-1, 2])
        var_671 = paddle.tensor.creation.full([196, 1], 32, dtype='float32')
        var_672 = paddle.tensor.creation.arange(end=28)
        var_673 = var_672.__add__(0.5)
        var_674 = var_673.__mul__(16)
        var_675 = paddle.tensor.creation.arange(end=28)
        var_676 = var_675.__add__(0.5)
        var_677 = var_676.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_677, var_674)
        var_678 = out[0]
        var_679 = out[1]
        var_680 = var_679.__sub__(40.0)
        var_681 = var_678.__sub__(40.0)
        var_682 = var_679.__add__(40.0)
        var_683 = var_678.__add__(40.0)
        var_684 = paddle.tensor.manipulation.stack(
            [var_680, var_681, var_682, var_683], axis=-1
        )
        var_685 = var_684.astype('float32')
        var_686 = paddle.tensor.manipulation.stack([var_679, var_678], axis=-1)
        var_687 = var_686.astype('float32')
        var_688 = var_685.reshape([-1, 4])
        var_689 = var_687.reshape([-1, 2])
        var_690 = paddle.tensor.creation.full([784, 1], 16, dtype='float32')
        var_691 = paddle.tensor.creation.arange(end=56)
        var_692 = var_691.__add__(0.5)
        var_693 = var_692.__mul__(8)
        var_694 = paddle.tensor.creation.arange(end=56)
        var_695 = var_694.__add__(0.5)
        var_696 = var_695.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_696, var_693)
        var_697 = out[0]
        var_698 = out[1]
        var_699 = var_698.__sub__(20.0)
        var_700 = var_697.__sub__(20.0)
        var_701 = var_698.__add__(20.0)
        var_702 = var_697.__add__(20.0)
        var_703 = paddle.tensor.manipulation.stack(
            [var_699, var_700, var_701, var_702], axis=-1
        )
        var_704 = var_703.astype('float32')
        var_705 = paddle.tensor.manipulation.stack([var_698, var_697], axis=-1)
        var_706 = var_705.astype('float32')
        var_707 = var_704.reshape([-1, 4])
        var_708 = var_706.reshape([-1, 2])
        var_709 = paddle.tensor.creation.full([3136, 1], 8, dtype='float32')
        var_710 = paddle.tensor.manipulation.concat([var_669, var_688, var_707])
        var_711 = paddle.tensor.manipulation.concat([var_670, var_689, var_708])
        var_712 = paddle.tensor.manipulation.concat([var_671, var_690, var_709])
        return var_710, var_711, var_669, var_688, var_707, var_712


class TestSIR157(unittest.TestCase):
    def setUp(self):
        self.inputs = ()
        self.net = SIR157()

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
