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
# model: configs^ppyolo^ppyolov2_r50vd_dcn_365e_coco_single_dy2st_train
# api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,method||__sub__,method||clip,method||__sub__,method||clip,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||clip,method||__sub__,method||__sub__,method||__mul__,method||clip,method||__add__,method||__sub__,method||__add__,method||__truediv__,api||paddle.nn.functional.loss.binary_cross_entropy_with_logits,method||__mul__
import unittest

import numpy as np

import paddle


class SIR88(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_876,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: False)
        var_877,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: False)
        var_878,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: False)
        var_879,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: False)
        var_880,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: False)
        var_881,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: True)
        var_882,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: True)
        var_883,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: True)
        var_884,  # (shape: [1, 3, 52, 52, 1], dtype: paddle.float32, stop_gradient: True)
    ):
        var_885 = paddle.tensor.math.maximum(var_877, var_881)
        var_886 = paddle.tensor.math.maximum(var_878, var_882)
        var_887 = paddle.tensor.math.minimum(var_879, var_883)
        var_888 = paddle.tensor.math.minimum(var_880, var_884)
        var_889 = var_887.__sub__(var_885)
        var_890 = var_889.clip(0)
        var_891 = var_888.__sub__(var_886)
        var_892 = var_891.clip(0)
        var_893 = var_890.__mul__(var_892)
        var_894 = var_879.__sub__(var_877)
        var_895 = var_880.__sub__(var_878)
        var_896 = var_894.__mul__(var_895)
        var_897 = var_896.clip(0)
        var_898 = var_883.__sub__(var_881)
        var_899 = var_884.__sub__(var_882)
        var_900 = var_898.__mul__(var_899)
        var_901 = var_900.clip(0)
        var_902 = var_897.__add__(var_901)
        var_903 = var_902.__sub__(var_893)
        var_904 = var_903.__add__(1e-09)
        var_905 = var_893.__truediv__(var_904)
        var_906 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(
            var_876, var_905, reduction='none'
        )
        var_907 = var_906.__mul__(1.0)
        return var_907


class TestSIR88(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 52, 52, 1], dtype=paddle.float32),
        )
        self.net = SIR88()

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
