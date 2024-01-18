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
# model: configs^cascade_rcnn^cascade_rcnn_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.tensor.search.topk,api||paddle.tensor.creation.full,method||__gt__,method||__lt__,api||paddle.tensor.logic.logical_and,api||paddle.tensor.creation.zeros_like,api||paddle.tensor.search.where,method||__ge__,api||paddle.tensor.creation.ones_like,api||paddle.tensor.search.where,method||max,method||__gt__,method||__eq__,api||paddle.tensor.logic.logical_and,method||cast,method||sum,method||__gt__,api||paddle.tensor.creation.ones_like,api||paddle.tensor.search.where,method||flatten,method||flatten
import unittest

import numpy as np

import paddle


class SIR47(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_804,  # (shape: [2, 220968], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.search.topk(var_804, k=1, axis=0)
        var_807 = out[0]
        var_808 = out[1]
        var_809 = paddle.tensor.creation.full([1, 220968], -1, dtype='int32')
        var_810 = var_807.__gt__(-1)
        var_811 = var_807.__lt__(0.3)
        var_812 = paddle.tensor.logic.logical_and(var_810, var_811)
        var_813 = paddle.tensor.creation.zeros_like(var_809)
        var_814 = paddle.tensor.search.where(var_812, var_813, var_809)
        var_815 = var_807.__ge__(0.7)
        var_816 = paddle.tensor.creation.ones_like(var_814)
        var_817 = paddle.tensor.search.where(var_815, var_816, var_814)
        var_818 = var_804.max(axis=1, keepdim=True)
        var_819 = var_804.__gt__(0)
        var_820 = var_804.__eq__(var_818)
        var_821 = paddle.tensor.logic.logical_and(var_819, var_820)
        var_822 = var_821.cast('int32')
        var_823 = var_822.sum(0, keepdim=True)
        var_824 = var_823.__gt__(0)
        var_825 = paddle.tensor.creation.ones_like(var_817)
        var_826 = paddle.tensor.search.where(var_824, var_825, var_817)
        var_827 = var_808.flatten()
        var_828 = var_826.flatten()
        return var_827, var_828


class TestSIR47(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[2, 220968], dtype=paddle.float32),)
        self.net = SIR47()

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
