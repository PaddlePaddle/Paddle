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
# api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.transpose,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,method||__getitem__,api||paddle.tensor.manipulation.gather,api||paddle.tensor.manipulation.concat,api||paddle.tensor.linalg.transpose,method||reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.linalg.matmul,method||__mul__,method||__add__,api||paddle.nn.functional.activation.softmax,api||paddle.tensor.linalg.matmul,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape
import unittest

import numpy as np

import paddle


class SIR59(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_895 = self.create_parameter(
            shape=[16, 49],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_890,  # (shape: [22, 16, 256], dtype: paddle.float32, stop_gradient: False)
        var_891,  # (shape: [22, 16, 49, 16], dtype: paddle.float32, stop_gradient: False)
        var_892,  # (shape: [22, 16, 49, 64], dtype: paddle.float32, stop_gradient: False)
        var_896,  # (shape: [16, 49], dtype: paddle.int64, stop_gradient: True)
    ):
        var_893 = paddle.tensor.manipulation.reshape(var_890, [22, 16, 16, 16])
        var_894 = paddle.tensor.linalg.transpose(var_893, perm=[0, 2, 1, 3])
        var_897 = paddle.tensor.linalg.transpose(self.var_895, (1, 0))
        var_898 = var_896.__getitem__(0)
        var_899 = paddle.tensor.manipulation.gather(var_897, var_898)
        var_900 = var_896.__getitem__(1)
        var_901 = paddle.tensor.manipulation.gather(var_897, var_900)
        var_902 = var_896.__getitem__(2)
        var_903 = paddle.tensor.manipulation.gather(var_897, var_902)
        var_904 = var_896.__getitem__(3)
        var_905 = paddle.tensor.manipulation.gather(var_897, var_904)
        var_906 = var_896.__getitem__(4)
        var_907 = paddle.tensor.manipulation.gather(var_897, var_906)
        var_908 = var_896.__getitem__(5)
        var_909 = paddle.tensor.manipulation.gather(var_897, var_908)
        var_910 = var_896.__getitem__(6)
        var_911 = paddle.tensor.manipulation.gather(var_897, var_910)
        var_912 = var_896.__getitem__(7)
        var_913 = paddle.tensor.manipulation.gather(var_897, var_912)
        var_914 = var_896.__getitem__(8)
        var_915 = paddle.tensor.manipulation.gather(var_897, var_914)
        var_916 = var_896.__getitem__(9)
        var_917 = paddle.tensor.manipulation.gather(var_897, var_916)
        var_918 = var_896.__getitem__(10)
        var_919 = paddle.tensor.manipulation.gather(var_897, var_918)
        var_920 = var_896.__getitem__(11)
        var_921 = paddle.tensor.manipulation.gather(var_897, var_920)
        var_922 = var_896.__getitem__(12)
        var_923 = paddle.tensor.manipulation.gather(var_897, var_922)
        var_924 = var_896.__getitem__(13)
        var_925 = paddle.tensor.manipulation.gather(var_897, var_924)
        var_926 = var_896.__getitem__(14)
        var_927 = paddle.tensor.manipulation.gather(var_897, var_926)
        var_928 = var_896.__getitem__(15)
        var_929 = paddle.tensor.manipulation.gather(var_897, var_928)
        var_930 = paddle.tensor.manipulation.concat(
            [
                var_899,
                var_901,
                var_903,
                var_905,
                var_907,
                var_909,
                var_911,
                var_913,
                var_915,
                var_917,
                var_919,
                var_921,
                var_923,
                var_925,
                var_927,
                var_929,
            ]
        )
        var_931 = paddle.tensor.linalg.transpose(var_930, (1, 0))
        var_932 = var_931.reshape((0, 16, 49))
        var_933 = paddle.tensor.linalg.transpose(var_891, perm=[0, 1, 3, 2])
        var_934 = paddle.tensor.linalg.matmul(var_894, var_933)
        var_935 = var_934.__mul__(0.25)
        var_936 = var_935.__add__(var_932)
        var_937 = paddle.nn.functional.activation.softmax(var_936)
        var_938 = paddle.tensor.linalg.matmul(var_937, var_892)
        var_939 = paddle.tensor.linalg.transpose(var_938, perm=[0, 2, 1, 3])
        var_940 = paddle.tensor.manipulation.reshape(var_939, [22, -1, 1024])
        return var_940


class TestSIR59(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 16, 256], dtype=paddle.float32),
            paddle.rand(shape=[22, 16, 49, 16], dtype=paddle.float32),
            paddle.rand(shape=[22, 16, 49, 64], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[16, 49], dtype=paddle.int64),
        )
        self.net = SIR59()

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
