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
# api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.split,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,api||paddle.tensor.manipulation.concat,api||paddle.tensor.linalg.transpose,method||reshape,api||paddle.tensor.linalg.matmul,method||__mul__,method||__add__,api||paddle.nn.functional.activation.softmax,api||paddle.tensor.linalg.matmul,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape
import unittest

import numpy as np

import paddle


class SIR45(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_718 = self.create_parameter(
            shape=[8, 49],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_709,  # (shape: [22, 49, 512], dtype: paddle.float32, stop_gradient: False)
        var_719,  # (shape: [49, 49], dtype: paddle.int64, stop_gradient: True)
    ):
        var_710 = paddle.tensor.manipulation.reshape(var_709, [22, 49, 8, 64])
        out = paddle.tensor.manipulation.split(var_710, [16, 16, 32], axis=3)
        var_711 = out[0]
        var_712 = out[1]
        var_713 = out[2]
        var_714 = paddle.tensor.linalg.transpose(var_711, perm=[0, 2, 1, 3])
        var_715 = paddle.tensor.linalg.transpose(var_712, perm=[0, 2, 1, 3])
        var_716 = paddle.tensor.linalg.transpose(var_713, perm=[0, 2, 1, 3])
        var_717 = paddle.tensor.linalg.transpose(var_715, perm=[0, 1, 3, 2])
        var_720 = paddle.tensor.linalg.transpose(self.var_718, (1, 0))
        var_721 = var_719.__getitem__(0)
        var_722 = paddle.tensor.manipulation.gather(var_720, var_721)
        var_723 = var_719.__getitem__(1)
        var_724 = paddle.tensor.manipulation.gather(var_720, var_723)
        var_725 = var_719.__getitem__(2)
        var_726 = paddle.tensor.manipulation.gather(var_720, var_725)
        var_727 = var_719.__getitem__(3)
        var_728 = paddle.tensor.manipulation.gather(var_720, var_727)
        var_729 = var_719.__getitem__(4)
        var_730 = paddle.tensor.manipulation.gather(var_720, var_729)
        var_731 = var_719.__getitem__(5)
        var_732 = paddle.tensor.manipulation.gather(var_720, var_731)
        var_733 = var_719.__getitem__(6)
        var_734 = paddle.tensor.manipulation.gather(var_720, var_733)
        var_735 = var_719.__getitem__(7)
        var_736 = paddle.tensor.manipulation.gather(var_720, var_735)
        var_737 = var_719.__getitem__(8)
        var_738 = paddle.tensor.manipulation.gather(var_720, var_737)
        var_739 = var_719.__getitem__(9)
        var_740 = paddle.tensor.manipulation.gather(var_720, var_739)
        var_741 = var_719.__getitem__(10)
        var_742 = paddle.tensor.manipulation.gather(var_720, var_741)
        var_743 = var_719.__getitem__(11)
        var_744 = paddle.tensor.manipulation.gather(var_720, var_743)
        var_745 = var_719.__getitem__(12)
        var_746 = paddle.tensor.manipulation.gather(var_720, var_745)
        var_747 = var_719.__getitem__(13)
        var_748 = paddle.tensor.manipulation.gather(var_720, var_747)
        var_749 = var_719.__getitem__(14)
        var_750 = paddle.tensor.manipulation.gather(var_720, var_749)
        var_751 = var_719.__getitem__(15)
        var_752 = paddle.tensor.manipulation.gather(var_720, var_751)
        var_753 = var_719.__getitem__(16)
        var_754 = paddle.tensor.manipulation.gather(var_720, var_753)
        var_755 = var_719.__getitem__(17)
        var_756 = paddle.tensor.manipulation.gather(var_720, var_755)
        var_757 = var_719.__getitem__(18)
        var_758 = paddle.tensor.manipulation.gather(var_720, var_757)
        var_759 = var_719.__getitem__(19)
        var_760 = paddle.tensor.manipulation.gather(var_720, var_759)
        var_761 = var_719.__getitem__(20)
        var_762 = paddle.tensor.manipulation.gather(var_720, var_761)
        var_763 = var_719.__getitem__(21)
        var_764 = paddle.tensor.manipulation.gather(var_720, var_763)
        var_765 = var_719.__getitem__(22)
        var_766 = paddle.tensor.manipulation.gather(var_720, var_765)
        var_767 = var_719.__getitem__(23)
        var_768 = paddle.tensor.manipulation.gather(var_720, var_767)
        var_769 = var_719.__getitem__(24)
        var_770 = paddle.tensor.manipulation.gather(var_720, var_769)
        var_771 = var_719.__getitem__(25)
        var_772 = paddle.tensor.manipulation.gather(var_720, var_771)
        var_773 = var_719.__getitem__(26)
        var_774 = paddle.tensor.manipulation.gather(var_720, var_773)
        var_775 = var_719.__getitem__(27)
        var_776 = paddle.tensor.manipulation.gather(var_720, var_775)
        var_777 = var_719.__getitem__(28)
        var_778 = paddle.tensor.manipulation.gather(var_720, var_777)
        var_779 = var_719.__getitem__(29)
        var_780 = paddle.tensor.manipulation.gather(var_720, var_779)
        var_781 = var_719.__getitem__(30)
        var_782 = paddle.tensor.manipulation.gather(var_720, var_781)
        var_783 = var_719.__getitem__(31)
        var_784 = paddle.tensor.manipulation.gather(var_720, var_783)
        var_785 = var_719.__getitem__(32)
        var_786 = paddle.tensor.manipulation.gather(var_720, var_785)
        var_787 = var_719.__getitem__(33)
        var_788 = paddle.tensor.manipulation.gather(var_720, var_787)
        var_789 = var_719.__getitem__(34)
        var_790 = paddle.tensor.manipulation.gather(var_720, var_789)
        var_791 = var_719.__getitem__(35)
        var_792 = paddle.tensor.manipulation.gather(var_720, var_791)
        var_793 = var_719.__getitem__(36)
        var_794 = paddle.tensor.manipulation.gather(var_720, var_793)
        var_795 = var_719.__getitem__(37)
        var_796 = paddle.tensor.manipulation.gather(var_720, var_795)
        var_797 = var_719.__getitem__(38)
        var_798 = paddle.tensor.manipulation.gather(var_720, var_797)
        var_799 = var_719.__getitem__(39)
        var_800 = paddle.tensor.manipulation.gather(var_720, var_799)
        var_801 = var_719.__getitem__(40)
        var_802 = paddle.tensor.manipulation.gather(var_720, var_801)
        var_803 = var_719.__getitem__(41)
        var_804 = paddle.tensor.manipulation.gather(var_720, var_803)
        var_805 = var_719.__getitem__(42)
        var_806 = paddle.tensor.manipulation.gather(var_720, var_805)
        var_807 = var_719.__getitem__(43)
        var_808 = paddle.tensor.manipulation.gather(var_720, var_807)
        var_809 = var_719.__getitem__(44)
        var_810 = paddle.tensor.manipulation.gather(var_720, var_809)
        var_811 = var_719.__getitem__(45)
        var_812 = paddle.tensor.manipulation.gather(var_720, var_811)
        var_813 = var_719.__getitem__(46)
        var_814 = paddle.tensor.manipulation.gather(var_720, var_813)
        var_815 = var_719.__getitem__(47)
        var_816 = paddle.tensor.manipulation.gather(var_720, var_815)
        var_817 = var_719.__getitem__(48)
        var_818 = paddle.tensor.manipulation.gather(var_720, var_817)
        var_819 = paddle.tensor.manipulation.concat(
            [
                var_722,
                var_724,
                var_726,
                var_728,
                var_730,
                var_732,
                var_734,
                var_736,
                var_738,
                var_740,
                var_742,
                var_744,
                var_746,
                var_748,
                var_750,
                var_752,
                var_754,
                var_756,
                var_758,
                var_760,
                var_762,
                var_764,
                var_766,
                var_768,
                var_770,
                var_772,
                var_774,
                var_776,
                var_778,
                var_780,
                var_782,
                var_784,
                var_786,
                var_788,
                var_790,
                var_792,
                var_794,
                var_796,
                var_798,
                var_800,
                var_802,
                var_804,
                var_806,
                var_808,
                var_810,
                var_812,
                var_814,
                var_816,
                var_818,
            ]
        )
        var_820 = paddle.tensor.linalg.transpose(var_819, (1, 0))
        var_821 = var_820.reshape((0, 49, 49))
        var_822 = paddle.tensor.linalg.matmul(var_714, var_717)
        var_823 = var_822.__mul__(0.25)
        var_824 = var_823.__add__(var_821)
        var_825 = paddle.nn.functional.activation.softmax(var_824)
        var_826 = paddle.tensor.linalg.matmul(var_825, var_716)
        var_827 = paddle.tensor.linalg.transpose(var_826, perm=[0, 2, 1, 3])
        var_828 = paddle.tensor.manipulation.reshape(var_827, [22, 49, 256])
        return var_828


class TestSIR45(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 49, 512], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[49, 49], dtype=paddle.int64),
        )
        self.net = SIR45()

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
