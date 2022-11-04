# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
import paddle
from paddle.sparse import nn
import paddle.sparse as sparse
import paddle.fluid as fluid
import copy


class TestSparseBatchNorm(unittest.TestCase):
    def test(self):
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        paddle.seed(0)
        channels = 4
        shape = [2, 3, 6, 6, channels]
        # there is no zero in dense_x
        dense_x = paddle.randn(shape)
        dense_x.stop_gradient = False

        batch_norm = paddle.nn.BatchNorm3D(channels, data_format="NDHWC")
        dense_y = batch_norm(dense_x)
        dense_y.backward(dense_y)

        sparse_dim = 4
        dense_x2 = copy.deepcopy(dense_x)
        dense_x2.stop_gradient = False
        sparse_x = dense_x2.to_sparse_coo(sparse_dim)
        sparse_batch_norm = paddle.sparse.nn.BatchNorm(channels)
        # set same params
        sparse_batch_norm._mean.set_value(batch_norm._mean)
        sparse_batch_norm._variance.set_value(batch_norm._variance)
        sparse_batch_norm.weight.set_value(batch_norm.weight)

        sparse_y = sparse_batch_norm(sparse_x)
        # compare the result with dense batch_norm
        assert np.allclose(
            dense_y.flatten().numpy(),
            sparse_y.values().flatten().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )

        # test backward
        sparse_y.backward(sparse_y)
        assert np.allclose(
            dense_x.grad.flatten().numpy(),
            sparse_x.grad.values().flatten().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

    def test_error_layout(self):
        with self.assertRaises(ValueError):
            shape = [2, 3, 6, 6, 3]
            x = paddle.randn(shape)
            sparse_x = x.to_sparse_coo(4)
            sparse_batch_norm = paddle.sparse.nn.BatchNorm(
                3, data_format='NCDHW'
            )
            sparse_batch_norm(sparse_x)

    def test2(self):
        paddle.seed(123)
        channels = 3
        x_data = paddle.randn((1, 6, 6, 6, channels)).astype('float32')
        dense_x = paddle.to_tensor(x_data)
        sparse_x = dense_x.to_sparse_coo(4)
        batch_norm = paddle.sparse.nn.BatchNorm(channels)
        batch_norm_out = batch_norm(sparse_x)
        dense_bn = paddle.nn.BatchNorm1D(channels)
        dense_x = dense_x.reshape((-1, dense_x.shape[-1]))
        dense_out = dense_bn(dense_x)
        assert np.allclose(dense_out.numpy(), batch_norm_out.values().numpy())
        # [1, 6, 6, 6, 3]


class TestSyncBatchNorm(unittest.TestCase):
    def test_sync_batch_norm(self):
        x = np.array(
            [[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]
        ).astype('float32')
        x = paddle.to_tensor(x)
        sparse_x = x.to_sparse_coo(len(x.shape) - 1)

        if paddle.is_compiled_with_cuda():
            sparse_sync_bn = nn.SyncBatchNorm(2)
            sparse_hidden = sparse_sync_bn(sparse_x)

            dense_sync_bn = paddle.nn.SyncBatchNorm(2)
            x = x.reshape((-1, x.shape[-1]))
            dense_hidden = dense_sync_bn(x)
            assert np.allclose(
                sparse_hidden.values().numpy(), dense_hidden.numpy()
            )

    def test_convert(self):
        base_model = paddle.nn.Sequential(
            nn.Conv3D(3, 5, 3), nn.BatchNorm(5), nn.BatchNorm(5)
        )

        model = paddle.nn.Sequential(
            nn.Conv3D(3, 5, 3),
            nn.BatchNorm(5),
            nn.BatchNorm(
                5,
                weight_attr=fluid.ParamAttr(name='bn.scale'),
                bias_attr=fluid.ParamAttr(name='bn.bias'),
            ),
        )
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        for idx, sublayer in enumerate(base_model.sublayers()):
            if isinstance(sublayer, nn.BatchNorm):
                self.assertEqual(isinstance(model[idx], nn.SyncBatchNorm), True)


class TestStatic(unittest.TestCase):
    def test(self):
        paddle.enable_static()
        indices = paddle.static.data(
            name='indices', shape=[4, 4], dtype='int32'
        )
        values = paddle.static.data(
            name='values', shape=[4, 1], dtype='float32'
        )
        channels = 1
        dense_shape = [1, 1, 3, 4, channels]
        sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)

        sparse_batch_norm = paddle.sparse.nn.BatchNorm(channels)
        sp_y = sparse_batch_norm(sp_x)
        out = sp_y.to_dense()

        exe = paddle.static.Executor()
        indices_data = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values_data = np.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
        bias_data = np.array([1.0]).astype('float32')
        weight_data = np.array([2.0]).astype('float32')
        mean_data = np.array([1.0]).astype('float32')
        variance_data = np.array([2.0]).astype('float32')

        fetch = exe.run(
            feed={
                'indices': indices_data,
                'values': values_data,
                'batch_norm_0.b_0': bias_data,
                'batch_norm_0.w_0': weight_data,
                'batch_norm_0.w_1': mean_data,
                'batch_norm_0.w_2': variance_data,
            },
            fetch_list=[out],
            return_numpy=True,
        )
        correct_out = np.array(
            [
                [
                    [
                        [[0.0], [-1.6832708], [0.0], [0.1055764]],
                        [[0.0], [0.0], [1.8944236], [0.0]],
                        [[0.0], [0.0], [0.0], [3.683271]],
                    ]
                ]
            ]
        ).astype('float32')
        np.testing.assert_allclose(correct_out, fetch[0], rtol=1e-5)
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
