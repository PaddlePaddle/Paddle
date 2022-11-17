# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.framework import _test_eager_guard
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle


class TestBatchNorm(unittest.TestCase):
    def test_name(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            with fluid.dygraph.guard(p):
                batch_norm1d = paddle.nn.BatchNorm1D(1, name="test")

    def test_error(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            # paddle.disable_static()
            x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
            x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')

            def error1d_dataformat():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                batch_norm1d = paddle.nn.BatchNorm1D(1, data_format='NCDHW')
                batch_norm1d(paddle.to_tensor(x_data_4))

            def error2d_dataformat():
                x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')
                batch_norm2d = paddle.nn.BatchNorm2D(1, data_format='NCDHW')
                batch_norm2d(paddle.to_tensor(x_data_3))

            def error3d_dataformat():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                batch_norm3d = paddle.nn.BatchNorm3D(1, data_format='NCL')
                batch_norm3d(paddle.to_tensor(x_data_4))

            def error1d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                batch_norm1d = paddle.nn.BatchNorm1D(1)
                batch_norm1d(paddle.to_tensor(x_data_4))

            def error2d():
                x_data_3 = np.random.random(size=(2, 1, 3)).astype('float32')
                batch_norm2d = paddle.nn.BatchNorm2D(1)
                batch_norm2d(paddle.to_tensor(x_data_3))

            def error3d():
                x_data_4 = np.random.random(size=(2, 1, 3, 3)).astype('float32')
                batch_norm3d = paddle.nn.BatchNorm3D(1)
                batch_norm3d(paddle.to_tensor(x_data_4))

            with fluid.dygraph.guard(p):
                self.assertRaises(ValueError, error1d)
                self.assertRaises(ValueError, error2d)
                self.assertRaises(ValueError, error3d)
                self.assertRaises(ValueError, error1d_dataformat)
                self.assertRaises(ValueError, error2d_dataformat)
                self.assertRaises(ValueError, error3d_dataformat)

    def test_large_batch(self):
        def compute_baseline(x):
            with fluid.dygraph.guard(p):
                bn = fluid.dygraph.BatchNorm(shape[1])
                x1 = paddle.to_tensor(x)
                x1.stop_gradient = False
                y = bn(x1)
                y.backward()
                return y.numpy(), x1.gradient()

        def compute_1d(x):
            with fluid.dygraph.guard(p):
                with _test_eager_guard():
                    bn = paddle.nn.BatchNorm1D(shape[1])
                    x1 = paddle.to_tensor(x)
                    x1.stop_gradient = False
                    y = bn(x1)
                    y.backward()
                    return y.numpy(), x1.gradient()

        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            # [N, C]
            shape = [200000, 4]
            x = np.random.randn(*shape).astype("float32")
            y1, g1 = compute_baseline(x)
            y2, g2 = compute_1d(x)
            np.testing.assert_allclose(g1, g2, rtol=1e-05)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

            # [N, C, L]
            shape = [1000000, 4, 4]
            x = np.random.randn(*shape).astype("float32")
            y1, g1 = compute_baseline(x)
            y2, g2 = compute_1d(x)
            np.testing.assert_allclose(g1, g2, rtol=1e-05)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_eager_api(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    bn = fluid.dygraph.BatchNorm(shape[1])
                    # bn = paddle.nn.BatchNorm2D(shape[1])
                    x1 = paddle.to_tensor(x)
                    x1.stop_gradient = False
                    y = bn(x1)
                    y.backward()
                    return y.numpy(), x1.gradient()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    with _test_eager_guard():
                        print("v2")
                        bn = paddle.nn.BatchNorm2D(shape[1])
                        x1 = paddle.to_tensor(x)
                        x1.stop_gradient = False
                        y = bn(x1)
                        y.backward()
                        return y.numpy(), x1.gradient()

            x = np.random.randn(*shape).astype("float32")
            y1, g1 = compute_v1(x)
            y2, g2 = compute_v2(x)
            np.testing.assert_allclose(g1, g2, rtol=1e-05)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [4, 10, 4, 4]

            def compute_v1(x, is_test, trainable_statistics):
                with fluid.dygraph.guard(p):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics,
                    )
                    y = bn(paddle.to_tensor(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.BatchNorm2D(shape[1])
                    y = bn(paddle.to_tensor(x))

                    with _test_eager_guard():
                        bn = paddle.nn.BatchNorm2D(shape[1])
                        eag_y = bn(paddle.to_tensor(x))
                    assert np.allclose(eag_y.numpy(), y.numpy())
                return y.numpy()

            def compute_v3(x, is_test, trainable_statistics):
                with fluid.dygraph.guard(p):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        param_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Constant(1.0),
                            trainable=False,
                        ),
                        bias_attr=fluid.ParamAttr(
                            initializer=fluid.initializer.Constant(0.0),
                            trainable=False,
                        ),
                        trainable_statistics=trainable_statistics,
                    )
                    y = bn(paddle.to_tensor(x))
                return y.numpy()

            def compute_v4(x):
                with fluid.dygraph.guard(p):
                    bn = paddle.nn.BatchNorm2D(
                        shape[1], weight_attr=False, bias_attr=False
                    )
                    y = bn(paddle.to_tensor(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x, False, False)
            y2 = compute_v2(x)
            y3 = compute_v3(x, False, False)
            y4 = compute_v4(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)
            np.testing.assert_allclose(y3, y4, rtol=1e-05)

    def test_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np, is_test, trainable_statistics):
                with program_guard(Program(), Program()):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics,
                    )
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = bn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    bn = paddle.nn.BatchNorm2D(shape[1])
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = bn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x, False, False)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


class TestBatchNormChannelLast(unittest.TestCase):
    def setUp(self):
        self.original_dtyep = paddle.get_default_dtype()
        # MIOPEN not support data type of double
        if core.is_compiled_with_rocm():
            paddle.set_default_dtype("float32")
        else:
            paddle.set_default_dtype("float64")
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def tearDown(self):
        paddle.set_default_dtype(self.original_dtyep)

    def test_1d(self):
        for p in self.places:
            with fluid.dygraph.guard(p):
                x = paddle.randn([2, 6, 4])
                net1 = paddle.nn.BatchNorm1D(4, data_format="NLC")
                net2 = paddle.nn.BatchNorm1D(4)
                net2.weight = net1.weight
                net2.bias = net1.bias
                y1 = net1(x)
                channel_first_x = paddle.transpose(x, [0, 2, 1])
                y2 = net2(channel_first_x)
                y2 = paddle.transpose(y2, [0, 2, 1])
                if core.is_compiled_with_rocm():
                    # HIP will fail if no atol
                    np.testing.assert_allclose(
                        y1.numpy(), y2.numpy(), rtol=1e-05, atol=1e-07
                    )
                else:
                    np.testing.assert_allclose(
                        y1.numpy(), y2.numpy(), rtol=1e-05
                    )

    def test_2d(self):
        for p in self.places:
            with fluid.dygraph.guard(p):
                x = paddle.randn([2, 6, 6, 4])
                net1 = paddle.nn.BatchNorm2D(4, data_format="NHWC")
                net2 = paddle.nn.BatchNorm2D(4)
                net2.weight = net1.weight
                net2.bias = net1.bias
                y1 = net1(x)
                channel_first_x = paddle.transpose(x, [0, 3, 1, 2])
                y2 = net2(channel_first_x)
                y2 = paddle.transpose(y2, [0, 2, 3, 1])
                if core.is_compiled_with_rocm():
                    # HIP will fail if no atol
                    np.testing.assert_allclose(
                        y1.numpy(), y2.numpy(), rtol=1e-05, atol=1e-07
                    )
                else:
                    np.testing.assert_allclose(
                        y1.numpy(), y2.numpy(), rtol=1e-05
                    )

    def test_3d(self):
        for p in self.places:
            with fluid.dygraph.guard(p):
                x = paddle.randn([2, 6, 6, 6, 4])
                net1 = paddle.nn.BatchNorm3D(4, data_format="NDHWC")
                net2 = paddle.nn.BatchNorm3D(4)
                net2.weight = net1.weight
                net2.bias = net1.bias
                y1 = net1(x)
                channel_first_x = paddle.transpose(x, [0, 4, 1, 2, 3])
                y2 = net2(channel_first_x)
                y2 = paddle.transpose(y2, [0, 2, 3, 4, 1])
                if core.is_compiled_with_rocm():
                    # HIP will fail if no atol
                    np.testing.assert_allclose(
                        y1.numpy(), y2.numpy(), rtol=1e-05, atol=1e-07
                    )
                else:
                    np.testing.assert_allclose(
                        y1.numpy(), y2.numpy(), rtol=1e-05
                    )

    def test_1d_opt(self):
        with fluid.dygraph.guard():
            batch_size = 13700
            channels = 16
            shape = (batch_size, channels)
            x = paddle.randn(shape)
            x_4d = x.reshape((batch_size, channels, 1, 1))

            x.stop_gradient = False
            x_4d.stop_gradient = False

            bn1d = paddle.nn.BatchNorm1D(channels)
            bn2d = paddle.nn.BatchNorm2D(channels)

            y = bn1d(x)
            y2 = bn2d(x_4d)

            y.backward()
            y2.backward()

            assert np.allclose(
                y.numpy().flatten(), y2.numpy().flatten(), atol=1e-5, rtol=1e-5
            )
            assert np.allclose(
                bn1d.weight.grad.numpy().flatten(),
                bn2d.weight.grad.numpy().flatten(),
                atol=1e-5,
                rtol=1e-5,
            )


class TestBatchNormUseGlobalStats(unittest.TestCase):
    def setUp(self):
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))
        self.init_test()

    # train mode
    def init_test(self):
        self.use_global_stats = True
        self.trainable_statistics = False

    def test_global_stats(self):
        for p in self.places:
            with fluid.dygraph.guard(p):
                x = paddle.randn([2, 6, 6, 4])
                net1 = paddle.fluid.dygraph.BatchNorm(
                    6,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Constant(1.0)
                    ),
                    use_global_stats=self.use_global_stats,
                    trainable_statistics=self.trainable_statistics,
                )
                net2 = paddle.nn.BatchNorm2D(
                    6, use_global_stats=self.use_global_stats
                )
                net2.weight = net1.weight
                net2.bias = net1.bias
                if self.trainable_statistics:
                    net1.training = False
                    net2.training = False
                y1 = net1(x)
                y2 = net2(x)
                np.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-05)


class TestBatchNormUseGlobalStatsCase1(TestBatchNormUseGlobalStats):
    # test mode
    def init_test(self):
        self.use_global_stats = False
        self.trainable_statistics = True


class TestBatchNormUseGlobalStatsCase2(TestBatchNormUseGlobalStats):
    # train mode
    def init_test(self):
        self.use_global_stats = False
        self.trainable_statistics = False


class TestBatchNormUseGlobalStatsCase3(TestBatchNormUseGlobalStats):
    # test mode
    def init_test(self):
        self.use_global_stats = True
        self.trainable_statistics = True


if __name__ == '__main__':
    import paddle

    paddle.enable_static()
    unittest.main()
