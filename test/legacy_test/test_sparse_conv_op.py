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

import logging
import unittest

import numpy as np
from utils import compare_legacy_with_pt

import paddle
import paddle.device
from paddle import sparse
from paddle.base import core
from paddle.base.framework import in_pir_mode

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


class TestSparseConv(unittest.TestCase):
    def test_conv2d(self):
        kernel = [[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]
        dense_kernel = paddle.to_tensor(
            kernel, dtype='float32', stop_gradient=False
        )
        dense_kernel = paddle.reshape(dense_kernel, [3, 3, 1, 1])
        paddings = [0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        bias = [1]

        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [1, 2, 3, 4]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[5], [11]]
        sparse_input = core.eager.sparse_coo_tensor(
            indices, values, dense_shape, False
        )
        out = paddle.sparse.nn.functional.conv2d(
            sparse_input,
            dense_kernel,
            bias=paddle.to_tensor(bias, dtype='float32'),
            stride=strides,
            padding=paddings,
            dilation=dilations,
            groups=1,
            data_format="NHWC",
        )
        out.backward(out)
        out = paddle.sparse.coalesce(out)
        np.testing.assert_array_equal(correct_out_values, out.values().numpy())

    def test_conv3d(self):
        kernel = [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
        dense_kernel = paddle.to_tensor(
            kernel, dtype='float32', stop_gradient=False
        )
        dense_kernel = paddle.reshape(dense_kernel, [1, 3, 3, 1, 1])
        paddings = [0, 0, 0]
        strides = [1, 1, 1]
        dilations = [1, 1, 1]
        bias = [1]

        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [1, 2, 3, 4]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[5], [11]]
        sparse_input = core.eager.sparse_coo_tensor(
            indices, values, dense_shape, False
        )
        out = paddle.sparse.nn.functional.conv3d(
            sparse_input,
            dense_kernel,
            bias=paddle.to_tensor(bias, dtype='float32'),
            stride=strides,
            padding=paddings,
            dilation=dilations,
            groups=1,
            data_format="NDHWC",
        )
        out.backward(out)
        out = paddle.sparse.coalesce(out)
        np.testing.assert_array_equal(correct_out_values, out.values().numpy())

    def test_subm_conv2d(self):
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        sparse_x = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, stop_gradient=True
        )
        weight = paddle.randn((1, 3, 3, 1), dtype='float32')
        y = paddle.sparse.nn.functional.subm_conv2d(
            sparse_x, weight, key='subm_conv'
        )
        np.testing.assert_array_equal(
            sparse_x.indices().numpy(), y.indices().numpy()
        )

    def test_subm_conv3d(self):
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        sparse_x = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, stop_gradient=True
        )
        weight = paddle.randn((1, 3, 3, 1, 1), dtype='float32')
        y = paddle.sparse.nn.functional.subm_conv3d(
            sparse_x, weight, key='subm_conv'
        )
        np.testing.assert_array_equal(
            sparse_x.indices().numpy(), y.indices().numpy()
        )

    def test_Conv2D(self):
        # (3, non_zero_num), 3-D:(N, H, W)
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        # (non_zero_num, C)
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )

        sparse_conv2d = paddle.sparse.nn.Conv2D(
            1, 1, (3, 3), data_format='NHWC'
        )
        sparse_out = sparse_conv2d(sparse_input)
        # test errors
        with self.assertRaises(ValueError):
            # Currently, only support data_format='NDHWC'
            conv2d = paddle.sparse.nn.SubmConv2D(
                1, 1, (3, 3), data_format='NCHW', key='subm_conv'
            )

    def test_Conv3D(self):
        # (4, non_zero_num), 4-D:(N, D, H, W)
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        # (non_zero_num, C)
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )

        sparse_conv3d = paddle.sparse.nn.Conv3D(
            1, 1, (1, 3, 3), data_format='NDHWC'
        )
        sparse_out = sparse_conv3d(sparse_input)
        # test errors
        with self.assertRaises(ValueError):
            # Currently, only support data_format='NDHWC'
            conv3d = paddle.sparse.nn.SubmConv3D(
                1, 1, (1, 3, 3), data_format='NCDHW', key='subm_conv'
            )

    def test_SubmConv2D(self):
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )

        subm_conv2d = paddle.sparse.nn.SubmConv2D(
            1, 1, (3, 3), data_format='NHWC', key='subm_conv'
        )
        # test extra_repr
        logger.info(subm_conv2d.extra_repr())

        sparse_out = subm_conv2d(sparse_input)
        # the output shape of subm_conv is same as input shape
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())

        # test errors
        with self.assertRaises(ValueError):
            # Currently, only support data_format='NHWC'
            conv2d = paddle.sparse.nn.SubmConv2D(
                1, 1, (3, 3), data_format='NCHW', key='subm_conv'
            )

    def test_SubmConv3D(self):
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )

        subm_conv3d = paddle.sparse.nn.SubmConv3D(
            1, 1, (1, 3, 3), data_format='NDHWC', key='subm_conv'
        )
        # test extra_repr
        print(subm_conv3d.extra_repr())

        sparse_out = subm_conv3d(sparse_input)
        # the output shape of subm_conv is same as input shape
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())

        # test errors
        with self.assertRaises(ValueError):
            # Currently, only support data_format='NDHWC'
            conv3d = paddle.sparse.nn.SubmConv3D(
                1, 1, (1, 3, 3), data_format='NCDHW', key='subm_conv'
            )

    def test_Conv2D_bias(self):
        paddle.seed(0)
        shape = [1, 4, 4, 3]
        x = paddle.randn(shape)
        sp_x = x.to_sparse_coo(3)
        conv2d = paddle.nn.Conv2D(3, 2, 3, data_format='NHWC')

        sp_conv2d = paddle.sparse.nn.Conv2D(3, 2, 3, data_format='NHWC')
        sp_conv2d.weight.set_value(
            paddle.to_tensor(conv2d.weight.numpy().transpose(2, 3, 1, 0))
        )
        sp_conv2d.bias.set_value(paddle.to_tensor(conv2d.bias.numpy()))

        x.stop_gradient = False
        out = conv2d(x)
        loss = out.mean()
        loss.backward()

        sp_x.stop_gradient = False
        sp_out = sp_conv2d(sp_x)
        dense_out = sp_out.to_dense()
        sp_loss = dense_out.mean()
        sp_loss.backward()
        np.testing.assert_allclose(
            out.numpy(), dense_out.numpy(), atol=1e-3, rtol=1e-3
        )
        np.testing.assert_allclose(
            conv2d.weight.grad.numpy().transpose(2, 3, 1, 0),
            sp_conv2d.weight.grad.numpy(),
            atol=1e-3,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            conv2d.bias.grad.numpy(),
            sp_conv2d.bias.grad.numpy(),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_Conv3D_bias(self):
        paddle.seed(0)
        shape = [1, 4, 4, 4, 3]
        x = paddle.randn(shape)
        sp_x = x.to_sparse_coo(4)
        conv3d = paddle.nn.Conv3D(3, 2, 3, data_format='NDHWC')

        sp_conv3d = paddle.sparse.nn.Conv3D(3, 2, 3, data_format='NDHWC')
        sp_conv3d.weight.set_value(
            paddle.to_tensor(conv3d.weight.numpy().transpose(2, 3, 4, 1, 0))
        )
        sp_conv3d.bias.set_value(paddle.to_tensor(conv3d.bias.numpy()))

        x.stop_gradient = False
        out = conv3d(x)
        loss = out.mean()
        loss.backward()

        sp_x.stop_gradient = False
        sp_out = sp_conv3d(sp_x)
        dense_out = sp_out.to_dense()
        sp_loss = dense_out.mean()
        sp_loss.backward()
        np.testing.assert_allclose(
            out.numpy(), dense_out.numpy(), atol=1e-3, rtol=1e-3
        )
        np.testing.assert_allclose(
            conv3d.weight.grad.numpy().transpose(2, 3, 4, 1, 0),
            sp_conv3d.weight.grad.numpy(),
            atol=1e-3,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            conv3d.bias.grad.numpy(),
            sp_conv3d.bias.grad.numpy(),
            atol=1e-5,
            rtol=1e-5,
        )


class TestStatic(unittest.TestCase):
    @compare_legacy_with_pt
    def test(self):
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(
                name='indices', shape=[4, 4], dtype='int32'
            )
            values = paddle.static.data(
                name='values', shape=[4, 1], dtype='float32'
            )
            dense_shape = [1, 1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)

            weight_shape = [1, 3, 3, 1, 1]
            weight = paddle.static.data(
                name='weight', shape=weight_shape, dtype='float32'
            )
            bias_shape = [1]
            bias = paddle.static.data(
                name='bias', shape=bias_shape, dtype='float32'
            )
            out = sparse.nn.functional.conv3d(
                sp_x,
                weight,
                bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                data_format="NDHWC",
            )
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()

            exe = paddle.static.Executor()

            indices_data = [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 2],
                [1, 3, 2, 3],
            ]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array(
                [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
            ).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')

            fetch = exe.run(
                feed={
                    'indices': indices_data,
                    'values': values_data,
                    'weight': weight_data,
                    'bias': bias_data,
                },
                fetch_list=[out, out_indices, out_values],
                return_numpy=True,
            )
            correct_out = np.array([[[[[5.0], [11.0]]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()

    @compare_legacy_with_pt
    def test_cpu(self):
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(
                name='indices', shape=[4, 4], dtype='int32'
            )
            values = paddle.static.data(
                name='values', shape=[4, 1], dtype='float32'
            )
            dense_shape = [1, 1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)

            weight_shape = [1, 3, 3, 1, 1]
            weight = paddle.static.data(
                name='weight', shape=weight_shape, dtype='float32'
            )
            bias_shape = [1]
            bias = paddle.static.data(
                name='bias', shape=bias_shape, dtype='float32'
            )
            out = sparse.nn.functional.conv3d(
                sp_x,
                weight,
                bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                data_format="NDHWC",
            )
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()

            place = paddle.CPUPlace()
            exe = paddle.static.Executor()

            indices_data = [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 2],
                [1, 3, 2, 3],
            ]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array(
                [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
            ).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')

            fetch = exe.run(
                feed={
                    'indices': indices_data,
                    'values': values_data,
                    'weight': weight_data,
                    'bias': bias_data,
                },
                fetch_list=[out, out_indices, out_values],
                return_numpy=True,
            )
            correct_out = np.array([[[[[5.0], [11.0]]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()

    @compare_legacy_with_pt
    def test2D(self):
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(
                name='indices', shape=[3, 4], dtype='int32'
            )
            values = paddle.static.data(
                name='values', shape=[4, 1], dtype='float32'
            )
            dense_shape = [1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)

            weight_shape = [3, 3, 1, 1]
            weight = paddle.static.data(
                name='weight', shape=weight_shape, dtype='float32'
            )
            bias_shape = [1]
            bias = paddle.static.data(
                name='bias', shape=bias_shape, dtype='float32'
            )
            out = sparse.nn.functional.conv2d(
                sp_x,
                weight,
                bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                data_format="NHWC",
            )
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()

            exe = paddle.static.Executor()

            indices_data = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array(
                [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
            ).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')

            fetch = exe.run(
                feed={
                    'indices': indices_data,
                    'values': values_data,
                    'weight': weight_data,
                    'bias': bias_data,
                },
                fetch_list=[out, out_indices, out_values],
                return_numpy=True,
            )
            correct_out = np.array([[[[5.0], [11.0]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()

    @compare_legacy_with_pt
    def test2D_cpu(self):
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(
                name='indices', shape=[3, 4], dtype='int32'
            )
            values = paddle.static.data(
                name='values', shape=[4, 1], dtype='float32'
            )
            dense_shape = [1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)

            weight_shape = [3, 3, 1, 1]
            weight = paddle.static.data(
                name='weight', shape=weight_shape, dtype='float32'
            )
            bias_shape = [1]
            bias = paddle.static.data(
                name='bias', shape=bias_shape, dtype='float32'
            )
            out = sparse.nn.functional.conv2d(
                sp_x,
                weight,
                bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                data_format="NHWC",
            )
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()

            place = paddle.CPUPlace()
            exe = paddle.static.Executor()

            indices_data = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array(
                [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
            ).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')

            fetch = exe.run(
                feed={
                    'indices': indices_data,
                    'values': values_data,
                    'weight': weight_data,
                    'bias': bias_data,
                },
                fetch_list=[out, out_indices, out_values],
                return_numpy=True,
            )
            correct_out = np.array([[[[5.0], [11.0]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()


devices = []
if paddle.device.get_device() != "cpu":
    devices.append(paddle.device.get_device())
else:
    devices.append('cpu')


class TestSparseSubmConvStatic(unittest.TestCase):
    '''
    test subm_conv2d and subm_conv3d in static graph in pir mode.
    compare the results of subm_conv2d in static graph and dynamic graph, use the result in dynamic graph as the correct answer.
    '''

    def check_result_subm_conv2d(self, x_shape, weight_shape):
        '''
        x_shape: the shape of input tensor x, [N, H, W, C]
        weight_shape: the shape of conv kernel, [kH, kW, C/g, M]
        compare the output of paddle.sparse.nn.functional.subm_conv2d in static graph and dynamic graph.
        '''
        for device in devices:
            paddle.device.set_device(device)
            x = paddle.rand(x_shape, dtype='float32')
            weight = paddle.randn(weight_shape, dtype='float32')
            x_indices_data, x_values_data = (
                x.detach().to_sparse_coo(sparse_dim=len(x_shape)).indices(),
                x.detach().to_sparse_coo(sparse_dim=len(x_shape)).values(),
            )
            w_indices_data, w_values_data = (
                weight.detach()
                .to_sparse_coo(sparse_dim=len(weight_shape))
                .indices(),
                weight.detach()
                .to_sparse_coo(sparse_dim=len(weight_shape))
                .values(),
            )
            x.stop_gradient = False
            weight.stop_gradient = False

            dynamic_out = paddle.sparse.nn.functional.subm_conv2d(x, weight)
            dynamic_out_dense = dynamic_out.to_dense()

            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_indices = paddle.static.data(
                    name="x_indices",
                    shape=x_indices_data.shape,
                    dtype=x_indices_data.dtype,
                )
                x_values = paddle.static.data(
                    name="x_values",
                    shape=x_values_data.shape,
                    dtype=x_values_data.dtype,
                )
                w_indices = paddle.static.data(
                    name="w_indices",
                    shape=w_indices_data.shape,
                    dtype=w_indices_data.dtype,
                )
                w_values = paddle.static.data(
                    name="w_values",
                    shape=w_values_data.shape,
                    dtype=w_values_data.dtype,
                )

                static_x = paddle.sparse.sparse_coo_tensor(
                    x_indices,
                    x_values,
                    shape=x_shape,
                    dtype=x.dtype,
                )
                static_w = paddle.sparse.sparse_coo_tensor(
                    w_indices,
                    w_values,
                    shape=weight_shape,
                    dtype=weight.dtype,
                )
                static_out = paddle.sparse.nn.functional.subm_conv2d(
                    static_x, static_w
                )
                static_dense_out = static_out.to_dense()

                st_exe = paddle.static.Executor()
                st_fetch = st_exe.run(
                    feed={
                        "x_indices": x_indices_data.numpy(),
                        "x_values": x_values_data.numpy(),
                        "w_indices": w_indices_data.numpy(),
                        "w_values": w_values_data.numpy(),
                    },
                    fetch_list=[static_dense_out],
                    return_numpy=True,
                )
                np.testing.assert_allclose(
                    dynamic_out_dense.numpy(), st_fetch[0], rtol=1e-05
                )
                paddle.disable_static()

    def check_result_subm_conv3d(self, x_shape, weight_shape):
        '''
        x_shape: the shape of input tensor x, [N, D, H, W, C]
        weight_shape: the shape of conv kernel, [kD, kH, kW, C/g, M]
        compare the output of paddle.sparse.nn.functional.subm_conv3d in static graph and dynamic graph.
        '''
        for device in devices:
            paddle.device.set_device(device)
            x = paddle.rand(x_shape, dtype='float32')
            weight = paddle.randn(weight_shape, dtype='float32')
            x_indices_data, x_values_data = (
                x.detach().to_sparse_coo(sparse_dim=len(x_shape)).indices(),
                x.detach().to_sparse_coo(sparse_dim=len(x_shape)).values(),
            )
            w_indices_data, w_values_data = (
                weight.detach()
                .to_sparse_coo(sparse_dim=len(weight_shape))
                .indices(),
                weight.detach()
                .to_sparse_coo(sparse_dim=len(weight_shape))
                .values(),
            )
            x.stop_gradient = False
            weight.stop_gradient = False

            dynamic_out = paddle.sparse.nn.functional.subm_conv3d(x, weight)
            dynamic_out_dense = dynamic_out.to_dense()

            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_indices = paddle.static.data(
                    name="x_indices",
                    shape=x_indices_data.shape,
                    dtype=x_indices_data.dtype,
                )
                x_values = paddle.static.data(
                    name="x_values",
                    shape=x_values_data.shape,
                    dtype=x_values_data.dtype,
                )
                w_indices = paddle.static.data(
                    name="w_indices",
                    shape=w_indices_data.shape,
                    dtype=w_indices_data.dtype,
                )
                w_values = paddle.static.data(
                    name="w_values",
                    shape=w_values_data.shape,
                    dtype=w_values_data.dtype,
                )

                static_x = paddle.sparse.sparse_coo_tensor(
                    x_indices,
                    x_values,
                    shape=x_shape,
                    dtype=x.dtype,
                )
                static_w = paddle.sparse.sparse_coo_tensor(
                    w_indices,
                    w_values,
                    shape=weight_shape,
                    dtype=weight.dtype,
                )
                static_out = paddle.sparse.nn.functional.subm_conv3d(
                    static_x, static_w
                )
                static_dense_out = static_out.to_dense()

                st_exe = paddle.static.Executor()
                st_fetch = st_exe.run(
                    feed={
                        "x_indices": x_indices_data.numpy(),
                        "x_values": x_values_data.numpy(),
                        "w_indices": w_indices_data.numpy(),
                        "w_values": w_values_data.numpy(),
                    },
                    fetch_list=[static_dense_out],
                    return_numpy=True,
                )
                np.testing.assert_allclose(
                    dynamic_out_dense.numpy(), st_fetch[0], rtol=1e-05
                )
                paddle.disable_static()

    def test_subm_conv2d(self):
        if in_pir_mode():
            self.check_result_subm_conv2d([1, 3, 4, 1], [3, 3, 1, 1])

    def test_subm_conv3d(self):
        if in_pir_mode():
            self.check_result_subm_conv3d([1, 1, 3, 4, 1], [1, 3, 3, 1, 1])


if __name__ == "__main__":
    unittest.main()
