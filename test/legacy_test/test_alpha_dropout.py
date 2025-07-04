#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core

# rename this function, or `pytest` will treat it as a fixture


class TestAlphaDropoutFunctionAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input = paddle.static.data(
                name="input", shape=[40, 40], dtype="float32"
            )
            res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.0)
            res2 = paddle.nn.functional.alpha_dropout(
                x=input, p=0.0, training=False
            )
            res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.0)

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np3 = np.zeros_like(in_np)

            exe = base.Executor(place)

            fetches = exe.run(
                main_prog,
                feed={"input": in_np},
                fetch_list=[res1, res2, res3],
            )
            np.testing.assert_allclose(fetches[0], res_np, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], res_np, rtol=1e-05)
            np.testing.assert_allclose(fetches[2], res_np3, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("float32")
                res_np = in_np
                res_np3 = np.zeros_like(in_np)
                input = paddle.to_tensor(in_np)
                input.stop_gradient = False

                res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.0)
                res2 = paddle.nn.functional.alpha_dropout(
                    x=input, p=0.0, training=False
                )
                res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.0)

                res_list = [res1, res2]
                for res in res_list:
                    np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)
                np.testing.assert_allclose(res3.numpy(), res_np3, rtol=1e-05)

                # test backward
                res1.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())

    def test_dygraph_bfp16(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with base.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("uint16")
                res_np = in_np
                res_np3 = np.zeros_like(in_np)
                input = paddle.to_tensor(in_np).astype("bfloat16")
                input.stop_gradient = False

                res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.0)
                res2 = paddle.nn.functional.alpha_dropout(
                    x=input, p=0.0, training=False
                )
                res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.0)

                res_list = [res1, res2]
                for res in res_list:
                    np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)
                np.testing.assert_allclose(res3.numpy(), res_np3, rtol=1e-05)

                # test backward
                res1.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())


class TestAlphaDropoutFunctionAPIError(unittest.TestCase):

    def test_input_type_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
                )
                paddle.nn.functional.alpha_dropout(x1, p=0.5)

            self.assertRaises(TypeError, test_Variable)

    def test_input_dtype_errors(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):

            def test_dtype():
                # the input dtype of dropout must be float32 or float64
                xr = paddle.static.data(
                    name='xr', shape=[3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.alpha_dropout(xr)

            self.assertRaises(TypeError, test_dtype)

            def test_pdtype():
                # p should be int or float
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.alpha_dropout(x2, p='0.5')

            self.assertRaises(TypeError, test_pdtype)

            def test_pvalue():
                # p should be 0.<=p<=1.
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.alpha_dropout(x2, p=1.2)

            self.assertRaises(ValueError, test_pvalue)


class TestAlphaDropoutClassAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("float32")
                result_np = input_np
                input = paddle.to_tensor(input_np)
                input.stop_gradient = False

                m = paddle.nn.AlphaDropout(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )

                # test backward
                result.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())

    def test_dygraph_bfp16(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with base.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("uint16")
                result_np = input_np
                input = paddle.to_tensor(input_np).astype("bfloat16")
                input.stop_gradient = False

                m = paddle.nn.AlphaDropout(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )

                # test backward
                result.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())

    def test_static_fp16_gpu(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([2, 3]).astype("float16")

                x = paddle.static.data(name="x", shape=[2, 3], dtype="float16")

                m = paddle.nn.AlphaDropout(p=0.0)
                y = m(x)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )

                np.testing.assert_allclose(res[0], input, rtol=1e-05)

    def test_static_bfp16_gpu(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([2, 3]).astype("uint16")

                x = paddle.static.data(name="x", shape=[2, 3], dtype="bfloat16")

                m = paddle.nn.AlphaDropout(p=0.0)
                y = m(x)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )

                np.testing.assert_allclose(res[0], input, rtol=1e-05)


class TestFeatureAlphaDropoutFunctionAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input = paddle.static.data(
                name="input", shape=[40, 40], dtype="float32"
            )
            res1 = paddle.nn.functional.feature_alpha_dropout(x=input, p=0.0)
            res2 = paddle.nn.functional.feature_alpha_dropout(
                x=input, p=0.0, training=False
            )
            res3 = paddle.nn.functional.feature_alpha_dropout(x=input, p=1.0)

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np3 = np.zeros_like(in_np)

            exe = base.Executor(place)

            fetches = exe.run(
                main_prog,
                feed={"input": in_np},
                fetch_list=[res1, res2, res3],
            )
            np.testing.assert_allclose(fetches[0], res_np, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], res_np, rtol=1e-05)
            np.testing.assert_allclose(fetches[2], res_np3, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("float32")
                res_np = in_np
                res_np3 = np.zeros_like(in_np)
                input = paddle.to_tensor(in_np)
                input.stop_gradient = False

                res1 = paddle.nn.functional.feature_alpha_dropout(
                    x=input, p=0.0
                )
                res2 = paddle.nn.functional.feature_alpha_dropout(
                    x=input, p=0.0, training=False
                )
                res3 = paddle.nn.functional.feature_alpha_dropout(
                    x=input, p=1.0
                )

                res_list = [res1, res2]
                for res in res_list:
                    np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)
                np.testing.assert_allclose(res3.numpy(), res_np3, rtol=1e-05)

                # test backward
                res1.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())

    def test_dygraph_bfp16(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with base.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("uint16")
                res_np = in_np
                res_np3 = np.zeros_like(in_np)
                input = paddle.to_tensor(in_np).astype("bfloat16")
                input.stop_gradient = False

                res1 = paddle.nn.functional.feature_alpha_dropout(
                    x=input, p=0.0
                )
                res2 = paddle.nn.functional.feature_alpha_dropout(
                    x=input, p=0.0, training=False
                )
                res3 = paddle.nn.functional.feature_alpha_dropout(
                    x=input, p=1.0
                )

                res_list = [res1, res2]
                for res in res_list:
                    np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)
                np.testing.assert_allclose(res3.numpy(), res_np3, rtol=1e-05)

                # test backward
                res1.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())


class TestFeatureAlphaDropoutFunctionAPIError(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_input_ndim_errors(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random(
                    [
                        40,
                    ]
                ).astype("float32")
                input = paddle.to_tensor(in_np)

                with self.assertRaises(ValueError):
                    _ = paddle.nn.functional.feature_alpha_dropout(
                        x=input, p=0.0
                    )

    def test_input_type_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
                )
                paddle.nn.functional.feature_alpha_dropout(x1, p=0.5)

            self.assertRaises(TypeError, test_Variable)

    def test_input_dtype_errors(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):

            def test_dtype():
                # the input dtype of dropout must be float32 or float64
                xr = paddle.static.data(
                    name='xr', shape=[3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.feature_alpha_dropout(xr)

            self.assertRaises(TypeError, test_dtype)

            def test_pdtype():
                # p should be int or float
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.feature_alpha_dropout(x2, p='0.5')

            self.assertRaises(TypeError, test_pdtype)

            def test_pvalue():
                # p should be 0.<=p<=1.
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.feature_alpha_dropout(x2, p=1.2)

            self.assertRaises(ValueError, test_pvalue)


class TestFeatureAlphaDropoutClassAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("float32")
                result_np = input_np
                input = paddle.to_tensor(input_np)
                input.stop_gradient = False

                m = paddle.nn.FeatureAlphaDropout(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )

                # test backward
                result.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())

    def test_dygraph_bfp16(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with base.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("uint16")
                result_np = input_np
                input = paddle.to_tensor(input_np).astype("bfloat16")
                input.stop_gradient = False

                m = paddle.nn.FeatureAlphaDropout(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )

                # test backward
                result.backward()
                grad = input.grad
                self.assertTrue(grad.dtype == input.dtype)
                self.assertTrue(grad.shape == input.shape)
                self.assertTrue((grad == 1).all())

    def test_static_fp16_gpu(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([2, 3]).astype("float16")

                x = paddle.static.data(name="x", shape=[2, 3], dtype="float16")

                m = paddle.nn.FeatureAlphaDropout(p=0.0)
                y = m(x)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )

                np.testing.assert_allclose(res[0], input, rtol=1e-05)

    def test_static_bfp16_gpu(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([2, 3]).astype("uint16")

                x = paddle.static.data(name="x", shape=[2, 3], dtype="bfloat16")

                m = paddle.nn.FeatureAlphaDropout(p=0.0)
                y = m(x)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )

                np.testing.assert_allclose(res[0], input, rtol=1e-05)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
