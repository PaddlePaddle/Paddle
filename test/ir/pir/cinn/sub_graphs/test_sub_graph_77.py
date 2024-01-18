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
# model: configs^rotate^ppyoloe_r^ppyoloe_r_crn_s_3x_dota_single_dy2st_train
# api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR169(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_714,  # (shape: [1, 384, 32, 32], dtype: paddle.float32, stop_gradient: False)
        var_715,  # (shape: [1, 192, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_716,  # (shape: [1, 96, 128, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_717 = paddle.tensor.attribute.shape(var_714)
        var_718 = var_717.__getitem__(0)
        var_719 = var_717.__getitem__(1)
        var_720 = var_717.__getitem__(2)
        var_721 = var_717.__getitem__(3)
        var_722 = paddle.tensor.creation.arange(end=var_721)
        var_723 = var_722.__add__(0.5)
        var_724 = var_723.__mul__(32)
        var_725 = paddle.tensor.creation.arange(end=var_720)
        var_726 = var_725.__add__(0.5)
        var_727 = var_726.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_727, var_724)
        var_728 = out[0]
        var_729 = out[1]
        var_730 = paddle.tensor.manipulation.stack([var_729, var_728], axis=-1)
        var_731 = paddle.tensor.manipulation.cast(var_730, dtype='float32')
        var_732 = var_731.reshape([1, -1, 2])
        var_733 = var_720.__mul__(var_721)
        var_734 = paddle.tensor.creation.full(
            [1, var_733, 1], 32, dtype='float32'
        )
        var_735 = var_720.__mul__(var_721)
        var_736 = paddle.tensor.attribute.shape(var_715)
        var_737 = var_736.__getitem__(0)
        var_738 = var_736.__getitem__(1)
        var_739 = var_736.__getitem__(2)
        var_740 = var_736.__getitem__(3)
        var_741 = paddle.tensor.creation.arange(end=var_740)
        var_742 = var_741.__add__(0.5)
        var_743 = var_742.__mul__(16)
        var_744 = paddle.tensor.creation.arange(end=var_739)
        var_745 = var_744.__add__(0.5)
        var_746 = var_745.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_746, var_743)
        var_747 = out[0]
        var_748 = out[1]
        var_749 = paddle.tensor.manipulation.stack([var_748, var_747], axis=-1)
        var_750 = paddle.tensor.manipulation.cast(var_749, dtype='float32')
        var_751 = var_750.reshape([1, -1, 2])
        var_752 = var_739.__mul__(var_740)
        var_753 = paddle.tensor.creation.full(
            [1, var_752, 1], 16, dtype='float32'
        )
        var_754 = var_739.__mul__(var_740)
        var_755 = paddle.tensor.attribute.shape(var_716)
        var_756 = var_755.__getitem__(0)
        var_757 = var_755.__getitem__(1)
        var_758 = var_755.__getitem__(2)
        var_759 = var_755.__getitem__(3)
        var_760 = paddle.tensor.creation.arange(end=var_759)
        var_761 = var_760.__add__(0.5)
        var_762 = var_761.__mul__(8)
        var_763 = paddle.tensor.creation.arange(end=var_758)
        var_764 = var_763.__add__(0.5)
        var_765 = var_764.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_765, var_762)
        var_766 = out[0]
        var_767 = out[1]
        var_768 = paddle.tensor.manipulation.stack([var_767, var_766], axis=-1)
        var_769 = paddle.tensor.manipulation.cast(var_768, dtype='float32')
        var_770 = var_769.reshape([1, -1, 2])
        var_771 = var_758.__mul__(var_759)
        var_772 = paddle.tensor.creation.full(
            [1, var_771, 1], 8, dtype='float32'
        )
        var_773 = var_758.__mul__(var_759)
        var_774 = paddle.tensor.manipulation.concat(
            [var_732, var_751, var_770], axis=1
        )
        var_775 = paddle.tensor.manipulation.concat(
            [var_734, var_753, var_772], axis=1
        )
        return var_774, var_735, var_754, var_773, var_775


class TestSIR169(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 384, 32, 32], dtype=paddle.float32),
            paddle.rand(shape=[1, 192, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 96, 128, 128], dtype=paddle.float32),
        )
        self.net = SIR169()

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
