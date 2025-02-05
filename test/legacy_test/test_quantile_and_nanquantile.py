#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

API_list = [
    (paddle.quantile, np.quantile),
    (paddle.nanquantile, np.nanquantile),
]


class TestQuantileAndNanquantile(unittest.TestCase):
    """
    This class is used for numerical precision testing. If there is
    a corresponding numpy API, the precision comparison can be performed directly.
    Otherwise, it needs to be verified by numpy implemented function.
    """

    def setUp(self):
        self.input_data = np.random.rand(4, 7, 6)

    # Test correctness when q and axis are set.
    def test_single_q(self):
        inp = self.input_data
        for func, res_func in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.5, axis=2)
            np_res = res_func(inp, q=0.5, axis=2)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 1, 2] = np.nan

    # Test correctness for default axis.
    def test_with_no_axis(self):
        inp = self.input_data
        for func, res_func in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.35)
            np_res = res_func(inp, q=0.35)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 2, 1] = np.nan
            inp[0, 1, 2] = np.nan

    # Test correctness for multiple axis.
    def test_with_multi_axis(self):
        inp = self.input_data
        for func, res_func in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.75, axis=[0, 2])
            np_res = res_func(inp, q=0.75, axis=[0, 2])
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 5, 3] = np.nan
            inp[0, 6, 2] = np.nan

    # Test correctness when keepdim is set.
    def test_with_keepdim(self):
        inp = self.input_data
        for func, res_func in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.35, axis=2, keepdim=True)
            np_res = res_func(inp, q=0.35, axis=2, keepdims=True)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 3, 4] = np.nan

    # Test correctness when all parameters are set.
    def test_with_keepdim_and_multiple_axis(self):
        inp = self.input_data
        for func, res_func in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.1, axis=[1, 2], keepdim=True)
            np_res = res_func(inp, q=0.1, axis=[1, 2], keepdims=True)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 6, 3] = np.nan

    # Test correctness when q = 0.
    def test_with_boundary_q(self):
        inp = self.input_data
        for func, res_func in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0, axis=1)
            np_res = res_func(inp, q=0, axis=1)
            np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)
            inp[0, 2, 5] = np.nan

    # Test correctness when input includes NaN.
    def test_quantile_include_NaN(self):
        input_data = np.random.randn(2, 3, 4)
        input_data[0, 1, 1] = np.nan
        x = paddle.to_tensor(input_data)
        paddle_res = paddle.quantile(x, q=0.35, axis=0)
        np_res = np.quantile(input_data, q=0.35, axis=0)
        np.testing.assert_allclose(
            paddle_res.numpy(), np_res, rtol=1e-05, equal_nan=True
        )

    # Test correctness when input filled with NaN.
    def test_nanquantile_all_NaN(self):
        input_data = np.full(shape=[2, 3], fill_value=np.nan)
        input_data[0, 2] = 0
        x = paddle.to_tensor(input_data)
        paddle_res = paddle.nanquantile(x, q=0.35, axis=0)
        np_res = np.nanquantile(input_data, q=0.35, axis=0)
        np.testing.assert_allclose(
            paddle_res.numpy(), np_res, rtol=1e-05, equal_nan=True
        )

    def test_interpolation(self):
        input_data = np.random.randn(2, 3, 4)
        input_data[0, 1, 1] = np.nan
        x = paddle.to_tensor(input_data)
        for op, ref_op in API_list:
            for mode in ["lower", "higher", "midpoint", "nearest"]:
                paddle_res = op(x, q=0.35, axis=0, interpolation=mode)
                np_res = ref_op(input_data, q=0.35, axis=0, method=mode)
                np.testing.assert_allclose(
                    paddle_res.numpy(), np_res, rtol=1e-05, equal_nan=True
                )

    def test_backward(self):
        def check_grad(x, q, axis, target_grad, apis=None):
            x = np.array(x, dtype="float32")
            paddle.disable_static()
            for op, _ in apis or API_list:
                x_p = paddle.to_tensor(x, dtype="float32", stop_gradient=False)
                op(x_p, q, axis).sum().backward()
                np.testing.assert_allclose(
                    x_p.grad.numpy(),
                    np.array(target_grad, dtype="float32"),
                    rtol=1e-05,
                    equal_nan=True,
                )
            paddle.enable_static()
            opt = paddle.optimizer.SGD(learning_rate=0.01)
            for op, _ in apis or API_list:
                s_p = paddle.static.Program()
                m_p = paddle.static.Program()
                with paddle.static.program_guard(m_p, s_p):
                    x_p = paddle.static.data(
                        name="x",
                        shape=x.shape,
                        dtype=paddle.float32,
                    )
                    x_p.stop_gradient = False
                    q_p = paddle.static.data(
                        name="q",
                        shape=[len(q)] if isinstance(q, list) else [],
                        dtype=paddle.float32,
                    )
                    loss = op(x_p, q_p, axis).sum()
                    opt.minimize(loss)
                    exe = paddle.static.Executor()
                    exe.run(paddle.static.default_startup_program())
                    if paddle.framework.use_pir_api():
                        o = exe.run(
                            paddle.static.default_main_program(),
                            feed={"x": x, "q": np.array(q, dtype="float32")},
                            fetch_list=[],
                        )
                    else:
                        o = exe.run(
                            paddle.static.default_main_program(),
                            feed={"x": x, "q": np.array(q, dtype="float32")},
                            fetch_list=["x@GRAD"],
                        )[0]
                        np.testing.assert_allclose(
                            o,
                            np.array(target_grad, dtype="float32"),
                            rtol=1e-05,
                            equal_nan=True,
                        )
            paddle.disable_static()

        check_grad([1, 2, 3], 0.5, 0, [0, 1, 0])
        check_grad(
            [1, 2, 3, 4] * 2, [0.55, 0.7], 0, [0, 0, 0.95, 0, 0, 0.15, 0.9, 0]
        )
        check_grad(
            [[1, 2, 3], [4, 5, 6]],
            [0.3, 0.7],
            1,
            [[0.4, 1.2, 0.4], [0.4, 1.2, 0.4]],
        )
        # quantile
        check_grad(
            [1, float("nan"), 3], 0.5, 0, [0, 1, 0], [(paddle.quantile, None)]
        )
        # nanquantile
        check_grad(
            [1, float("nan"), 3],
            0.5,
            0,
            [0.5, 0, 0.5],
            [(paddle.nanquantile, None)],
        )


class TestMuitlpleQ(unittest.TestCase):
    """
    This class is used to test multiple input of q.
    """

    def setUp(self):
        self.input_data = np.random.rand(5, 3, 4)

    def test_quantile(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=[0.3, 0.44], axis=-2)
        np_res = np.quantile(self.input_data, q=[0.3, 0.44], axis=-2)
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

    def test_quantile_multiple_axis(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=[0.2, 0.67], axis=[1, -1])
        np_res = np.quantile(self.input_data, q=[0.2, 0.67], axis=[1, -1])
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

    def test_quantile_multiple_axis_keepdim(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(
            x, q=[0.1, 0.2, 0.3], axis=[1, 2], keepdim=True
        )
        np_res = np.quantile(
            self.input_data, q=[0.1, 0.2, 0.3], axis=[1, 2], keepdims=True
        )
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

    def test_quantile_with_tensor_input(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(
            x, q=paddle.to_tensor([0.1, 0.2]), axis=[1, 2], keepdim=True
        )
        np_res = np.quantile(
            self.input_data, q=[0.1, 0.2], axis=[1, 2], keepdims=True
        )
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)

    def test_quantile_with_zero_dim_tensor_input(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(
            x, q=paddle.to_tensor(0.1), axis=[1, 2], keepdim=True
        )
        np_res = np.quantile(self.input_data, q=0.1, axis=[1, 2], keepdims=True)
        np.testing.assert_allclose(paddle_res.numpy(), np_res, rtol=1e-05)


class TestError(unittest.TestCase):
    """
    This class is used to test that exceptions are thrown correctly.
    Validity of all parameter values and types should be considered.
    """

    def setUp(self):
        self.x = paddle.randn((2, 3, 4))

    def test_errors(self):
        # Test error when q > 1
        def test_q_range_error_1():
            paddle_res = paddle.quantile(self.x, q=1.5)

        self.assertRaises(ValueError, test_q_range_error_1)

        # Test error when q < 0
        def test_q_range_error_2():
            paddle_res = paddle.quantile(self.x, q=[0.2, -0.3])

        self.assertRaises(ValueError, test_q_range_error_2)

        # Test error with no valid q
        def test_q_range_error_3():
            paddle_res = paddle.quantile(self.x, q=[])

        self.assertRaises(ValueError, test_q_range_error_3)

        # Test error when x is not Tensor
        def test_x_type_error():
            x = [1, 3, 4]
            paddle_res = paddle.quantile(x, q=0.9)

        self.assertRaises(TypeError, test_x_type_error)

        # Test error when scalar axis is not int
        def test_axis_type_error_1():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=0.4)

        self.assertRaises(ValueError, test_axis_type_error_1)

        # Test error when axis in List is not int
        def test_axis_type_error_2():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, 0.4])

        self.assertRaises(ValueError, test_axis_type_error_2)

        # Test error when axis not in [-D, D)
        def test_axis_value_error_1():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=10)

        self.assertRaises(ValueError, test_axis_value_error_1)

        # Test error when axis not in [-D, D)
        def test_axis_value_error_2():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, -10])

        self.assertRaises(ValueError, test_axis_value_error_2)

        # Test error when q is not a 1-D tensor
        def test_tensor_input_1():
            paddle_res = paddle.quantile(
                self.x, q=paddle.randn((2, 3)), axis=[1, -10]
            )

        self.assertRaises(ValueError, test_tensor_input_1)

        def test_type_q():
            paddle_res = paddle.quantile(self.x, q={1}, axis=[1, -10])

        self.assertRaises(TypeError, test_type_q)

        def test_interpolation():
            paddle_res = paddle.quantile(
                self.x, q={1}, axis=[1, -10], interpolation=" "
            )

        self.assertRaises(TypeError, test_interpolation)


class TestQuantileRuntime(unittest.TestCase):
    """
    This class is used to test the API could run correctly with
    different devices, different data types, and dygraph/static graph mode.
    """

    def setUp(self):
        self.input_data = np.random.rand(4, 7)
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        if paddle.device.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_dygraph(self):
        paddle.disable_static()
        for func, res_func in API_list:
            for device in self.devices:
                # Check different devices
                paddle.set_device(device)
                for dtype in self.dtypes:
                    # Check different dtypes
                    np_input_data = self.input_data.astype(dtype)
                    x = paddle.to_tensor(np_input_data, dtype=dtype)
                    paddle_res = func(x, q=0.5, axis=1)
                    np_res = res_func(np_input_data, q=0.5, axis=1)
                    np.testing.assert_allclose(
                        paddle_res.numpy(), np_res, rtol=1e-05
                    )

    def test_static(self):
        paddle.enable_static()
        for func, res_func in API_list:
            for device in self.devices:
                x = paddle.static.data(
                    name="x", shape=self.input_data.shape, dtype="float32"
                )
                x_fp64 = paddle.static.data(
                    name="x_fp64",
                    shape=self.input_data.shape,
                    dtype="float64",
                )

                results = func(x, q=0.5, axis=1)
                np_input_data = self.input_data.astype("float32")
                results_fp64 = func(x_fp64, q=0.5, axis=1)
                np_input_data_fp64 = self.input_data.astype("float64")

                exe = paddle.static.Executor(device)
                paddle_res, paddle_res_fp64 = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": np_input_data, "x_fp64": np_input_data_fp64},
                    fetch_list=[results, results_fp64],
                )
                np_res = res_func(np_input_data, q=0.5, axis=1)
                np_res_fp64 = res_func(np_input_data_fp64, q=0.5, axis=1)
                np.testing.assert_allclose(paddle_res, np_res, rtol=1e-05)
                np.testing.assert_allclose(
                    paddle_res_fp64, np_res_fp64, rtol=1e-05
                )

    def test_static_tensor(self):
        paddle.enable_static()
        for func, res_func in API_list:
            s_p = paddle.static.Program()
            m_p = paddle.static.Program()
            with paddle.static.program_guard(m_p, s_p):
                for device in self.devices:
                    x = paddle.static.data(
                        name="x",
                        shape=self.input_data.shape,
                        dtype=paddle.float32,
                    )
                    q = paddle.static.data(
                        name="q", shape=(3,), dtype=paddle.float32
                    )
                    x_fp64 = paddle.static.data(
                        name="x_fp64",
                        shape=self.input_data.shape,
                        dtype=paddle.float64,
                    )

                    results = func(x, q=q, axis=1)
                    np_input_data = self.input_data.astype("float32")
                    results_fp64 = func(x_fp64, q=q, axis=1)
                    np_input_data_fp64 = self.input_data.astype("float64")
                    q_data = np.array([0.5, 0.5, 0.5]).astype("float32")

                    exe = paddle.static.Executor(device)
                    paddle_res, paddle_res_fp64 = exe.run(
                        paddle.static.default_main_program(),
                        feed={
                            "x": np_input_data,
                            "x_fp64": np_input_data_fp64,
                            "q": q_data,
                        },
                        fetch_list=[results, results_fp64],
                    )
                    np_res = res_func(np_input_data, q=[0.5, 0.5, 0.5], axis=1)
                    np_res_fp64 = res_func(
                        np_input_data_fp64, q=[0.5, 0.5, 0.5], axis=1
                    )
                    np.testing.assert_allclose(paddle_res, np_res, rtol=1e-05)
                    np.testing.assert_allclose(
                        paddle_res_fp64, np_res_fp64, rtol=1e-05
                    )

    def test_static_0d_tensor(self):
        paddle.enable_static()
        for func, res_func in API_list:
            for device in self.devices:
                s_p = paddle.static.Program()
                m_p = paddle.static.Program()
                with paddle.static.program_guard(m_p, s_p):
                    x = paddle.static.data(
                        name="x",
                        shape=self.input_data.shape,
                        dtype=paddle.float32,
                    )
                    q = paddle.static.data(
                        name="q", shape=[], dtype=paddle.float32
                    )
                    x_fp64 = paddle.static.data(
                        name="x_fp64",
                        shape=self.input_data.shape,
                        dtype=paddle.float64,
                    )

                    results = func(x, q=q, axis=1)
                    np_input_data = self.input_data.astype("float32")
                    results_fp64 = func(x_fp64, q=q, axis=1)
                    np_input_data_fp64 = self.input_data.astype("float64")
                    q_data = np.array(0.3).astype("float32")

                    exe = paddle.static.Executor(device)
                    paddle_res, paddle_res_fp64 = exe.run(
                        paddle.static.default_main_program(),
                        feed={
                            "x": np_input_data,
                            "x_fp64": np_input_data_fp64,
                            "q": q_data,
                        },
                        fetch_list=[results, results_fp64],
                    )
                    np_res = res_func(np_input_data, q=0.3, axis=1)
                    np_res_fp64 = res_func(np_input_data_fp64, q=0.3, axis=1)
                    np.testing.assert_allclose(paddle_res, np_res, rtol=1e-05)
                    np.testing.assert_allclose(
                        paddle_res_fp64, np_res_fp64, rtol=1e-05
                    )


if __name__ == '__main__':
    unittest.main()
