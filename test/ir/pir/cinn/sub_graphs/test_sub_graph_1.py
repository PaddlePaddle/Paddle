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
# model: ppcls^configs^ImageNet^Distillation^resnet34_distill_resnet18_afd
# method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,method||pow,method||mean,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,method||reshape,api||paddle.tensor.manipulation.stack
import unittest

import numpy as np

import paddle


class SIR146(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1369,  # (shape: [22, 64, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_1370,  # (shape: [22, 64, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_1371,  # (shape: [22, 128, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1372,  # (shape: [22, 128, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1373,  # (shape: [22, 256, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_1374,  # (shape: [22, 256, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_1375,  # (shape: [22, 512, 7, 7], dtype: paddle.float32, stop_gradient: False)
        var_1376,  # (shape: [22, 512, 7, 7], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1377 = var_1369.pow(2)
        var_1378 = var_1377.mean(1, keepdim=True)
        var_1379 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1378, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1380 = var_1379.reshape([22, 49])
        var_1381 = var_1370.pow(2)
        var_1382 = var_1381.mean(1, keepdim=True)
        var_1383 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1382, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1384 = var_1383.reshape([22, 49])
        var_1385 = var_1371.pow(2)
        var_1386 = var_1385.mean(1, keepdim=True)
        var_1387 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1386, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1388 = var_1387.reshape([22, 49])
        var_1389 = var_1372.pow(2)
        var_1390 = var_1389.mean(1, keepdim=True)
        var_1391 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1390, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1392 = var_1391.reshape([22, 49])
        var_1393 = var_1373.pow(2)
        var_1394 = var_1393.mean(1, keepdim=True)
        var_1395 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1394, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1396 = var_1395.reshape([22, 49])
        var_1397 = var_1374.pow(2)
        var_1398 = var_1397.mean(1, keepdim=True)
        var_1399 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1398, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1400 = var_1399.reshape([22, 49])
        var_1401 = var_1375.pow(2)
        var_1402 = var_1401.mean(1, keepdim=True)
        var_1403 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1402, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1404 = var_1403.reshape([22, 49])
        var_1405 = var_1376.pow(2)
        var_1406 = var_1405.mean(1, keepdim=True)
        var_1407 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1406, output_size=(7, 7), data_format='NCHW', name=None
        )
        var_1408 = var_1407.reshape([22, 49])
        var_1409 = paddle.tensor.manipulation.stack(
            [
                var_1380,
                var_1384,
                var_1388,
                var_1392,
                var_1396,
                var_1400,
                var_1404,
                var_1408,
            ],
            axis=1,
        )
        return var_1409


class TestSIR146(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 64, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[22, 64, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[22, 128, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 128, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 256, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 256, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 512, 7, 7], dtype=paddle.float32),
            paddle.rand(shape=[22, 512, 7, 7], dtype=paddle.float32),
        )
        self.net = SIR146()

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
