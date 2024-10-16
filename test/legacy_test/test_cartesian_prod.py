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
import os
import random
import unittest
from itertools import product

import numpy as np

import paddle
from paddle.base import core

paddle.enable_static()


class TestCartesianProdAPIBase(unittest.TestCase):
    def setUp(self):
        self.init_setting()
        self.a_shape = [random.randint(1, 5)]
        self.b_shape = [random.randint(1, 5)]
        self.c_shape = [random.randint(1, 5)]
        self.d_shape = [0]
        self.a_np = np.random.random(self.a_shape).astype(self.dtype_np)
        self.b_np = np.random.random(self.b_shape).astype(self.dtype_np)
        self.c_np = np.random.random(self.c_shape).astype(self.dtype_np)
        self.d_np = np.empty(0, self.dtype_np)

        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.place.append('cpu')
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')

    def init_setting(self):
        self.dtype_np = 'float32'

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        for place in self.place:
            with paddle.static.program_guard(main_program, startup_program):
                a = paddle.static.data(
                    name="a", shape=self.a_shape, dtype=self.dtype_np
                )
                b = paddle.static.data(
                    name="b", shape=self.b_shape, dtype=self.dtype_np
                )
                c = paddle.static.data(
                    name="c", shape=self.c_shape, dtype=self.dtype_np
                )
                d = paddle.static.data(
                    name="d", shape=self.d_shape, dtype=self.dtype_np
                )
                out1 = paddle.cartesian_prod([a, b, c])
                out2 = paddle.cartesian_prod([a, b, c, d])
                out3 = paddle.cartesian_prod([a])
                exe = paddle.static.Executor(place=place)
                feed_list = {
                    "a": self.a_np,
                    "b": self.b_np,
                    "c": self.c_np,
                    "d": self.d_np,
                }
                pd_res = exe.run(
                    main_program,
                    feed=feed_list,
                    fetch_list=[out1, out2, out3],
                )

                ref_res = np.array(
                    list(product(self.a_np, self.b_np, self.c_np))
                )
                np.testing.assert_allclose(ref_res, pd_res[0])
                # test empty
                ref_res = np.array(
                    list(product(self.a_np, self.b_np, self.c_np, self.d_np))
                ).reshape([0, 4])
                np.testing.assert_allclose(ref_res, pd_res[1])
                ref_res = np.array(list(product(self.a_np)))
                np.testing.assert_allclose(ref_res.flatten(), pd_res[2])

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            a = paddle.to_tensor(self.a_np)
            b = paddle.to_tensor(self.b_np)
            c = paddle.to_tensor(self.c_np)
            d = paddle.to_tensor(self.d_np)

            pd_res1 = paddle.cartesian_prod([a, b, c])
            ref_res = np.array(list(product(self.a_np, self.b_np, self.c_np)))
            np.testing.assert_allclose(ref_res, pd_res1)
            # test empty
            pd_res2 = paddle.cartesian_prod([a, b, c, d])
            ref_res = np.array(
                list(product(self.a_np, self.b_np, self.c_np, self.d_np))
            ).reshape([0, 4])
            np.testing.assert_allclose(ref_res, pd_res2)

            pd_res3 = paddle.cartesian_prod([a])
            ref_res = np.array(list(product(self.a_np)))
            np.testing.assert_allclose(ref_res.flatten(), pd_res3)


class TestCartesianProdErrors(unittest.TestCase):
    def test_errors(self):
        def test_input_not_1D():
            data_np = np.random.random((10, 10)).astype(np.float32)
            data_tensor = [paddle.to_tensor(data_np)]
            res = paddle.cartesian_prod(data_tensor)

        self.assertRaises(ValueError, test_input_not_1D)


class TestCartesianProdAPI1(TestCartesianProdAPIBase):
    def init_setting(self):
        self.dtype_np = 'int32'


class TestCartesianProdAPI2(TestCartesianProdAPIBase):
    def init_setting(self):
        self.dtype_np = 'int64'


class TestCartesianProdAPI3(TestCartesianProdAPIBase):
    def init_setting(self):
        self.dtype_np = 'float64'


class TestCartesianProdAPI4(TestCartesianProdAPIBase):
    def init_setting(self):
        self.dtype_np = 'complex64'


class TestCartesianProdAPI5(TestCartesianProdAPIBase):
    def init_setting(self):
        self.dtype_np = 'complex128'


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the float16",
)
class TestCartesianProdAPIFP16(unittest.TestCase):
    def setUp(self):
        self.dtype_np = 'float16'
        self.a_shape = [random.randint(1, 5)]
        self.b_shape = [random.randint(1, 5)]
        self.c_shape = [random.randint(1, 5)]
        self.d_shape = [0]
        self.a_np = np.random.random(self.a_shape).astype(self.dtype_np)
        self.b_np = np.random.random(self.b_shape).astype(self.dtype_np)
        self.c_np = np.random.random(self.c_shape).astype(self.dtype_np)
        self.d_np = np.empty(0, self.dtype_np)
        self.place = core.CUDAPlace(0)

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.static.data(
                name="a", shape=self.a_shape, dtype=self.dtype_np
            )
            b = paddle.static.data(
                name="b", shape=self.b_shape, dtype=self.dtype_np
            )
            c = paddle.static.data(
                name="c", shape=self.c_shape, dtype=self.dtype_np
            )
            d = paddle.static.data(
                name="d", shape=self.d_shape, dtype=self.dtype_np
            )
            out1 = paddle.cartesian_prod([a, b, c])
            out2 = paddle.cartesian_prod([a, b, c, d])
            out3 = paddle.cartesian_prod([a])
            exe = paddle.static.Executor(place=self.place)
            feed_list = {
                "a": self.a_np,
                "b": self.b_np,
                "c": self.c_np,
                "d": self.d_np,
            }
            pd_res = exe.run(
                main_program,
                feed=feed_list,
                fetch_list=[out1, out2, out3],
            )

            ref_res = np.array(list(product(self.a_np, self.b_np, self.c_np)))
            np.testing.assert_allclose(ref_res, pd_res[0])
            # test empty
            ref_res = np.array(
                list(product(self.a_np, self.b_np, self.c_np, self.d_np))
            ).reshape([0, 4])
            np.testing.assert_allclose(ref_res, pd_res[1])
            ref_res = np.array(list(product(self.a_np)))
            np.testing.assert_allclose(ref_res.flatten(), pd_res[2])

    def test_dygraph(self):
        paddle.disable_static(self.place)
        a = paddle.to_tensor(self.a_np)
        b = paddle.to_tensor(self.b_np)
        c = paddle.to_tensor(self.c_np)
        d = paddle.to_tensor(self.d_np)

        pd_res1 = paddle.cartesian_prod([a, b, c])
        ref_res = np.array(list(product(self.a_np, self.b_np, self.c_np)))
        np.testing.assert_allclose(ref_res, pd_res1)
        # test empty
        pd_res2 = paddle.cartesian_prod([a, b, c, d])
        ref_res = np.array(
            list(product(self.a_np, self.b_np, self.c_np, self.d_np))
        ).reshape([0, 4])
        np.testing.assert_allclose(ref_res, pd_res2)

        pd_res3 = paddle.cartesian_prod([a])
        ref_res = np.array(list(product(self.a_np)))
        np.testing.assert_allclose(ref_res.flatten(), pd_res3)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestCartesianProdAPIBF16(unittest.TestCase):
    def setUp(self):
        self.dtype_np = 'uint16'
        self.a_shape = [random.randint(1, 5)]
        self.b_shape = [random.randint(1, 5)]
        self.c_shape = [random.randint(1, 5)]
        self.d_shape = [0]
        self.a_np = np.random.random(self.a_shape).astype(self.dtype_np)
        self.b_np = np.random.random(self.b_shape).astype(self.dtype_np)
        self.c_np = np.random.random(self.c_shape).astype(self.dtype_np)
        self.d_np = np.empty(0, self.dtype_np)
        self.place = core.CUDAPlace(0)

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.static.data(
                name="a", shape=self.a_shape, dtype=self.dtype_np
            )
            b = paddle.static.data(
                name="b", shape=self.b_shape, dtype=self.dtype_np
            )
            c = paddle.static.data(
                name="c", shape=self.c_shape, dtype=self.dtype_np
            )
            d = paddle.static.data(
                name="d", shape=self.d_shape, dtype=self.dtype_np
            )
            out1 = paddle.cartesian_prod([a, b, c])
            out2 = paddle.cartesian_prod([a, b, c, d])
            out3 = paddle.cartesian_prod([a])
            exe = paddle.static.Executor(place=self.place)
            feed_list = {
                "a": self.a_np,
                "b": self.b_np,
                "c": self.c_np,
                "d": self.d_np,
            }
            pd_res = exe.run(
                main_program,
                feed=feed_list,
                fetch_list=[out1, out2, out3],
            )

            ref_res = np.array(list(product(self.a_np, self.b_np, self.c_np)))
            np.testing.assert_allclose(ref_res, pd_res[0])
            # test empty
            ref_res = np.array(
                list(product(self.a_np, self.b_np, self.c_np, self.d_np))
            ).reshape([0, 4])
            np.testing.assert_allclose(ref_res, pd_res[1])
            ref_res = np.array(list(product(self.a_np)))
            np.testing.assert_allclose(ref_res.flatten(), pd_res[2])

    def test_dygraph(self):
        paddle.disable_static(self.place)
        a = paddle.to_tensor(self.a_np)
        b = paddle.to_tensor(self.b_np)
        c = paddle.to_tensor(self.c_np)
        d = paddle.to_tensor(self.d_np)

        pd_res1 = paddle.cartesian_prod([a, b, c])
        ref_res = np.array(list(product(self.a_np, self.b_np, self.c_np)))
        np.testing.assert_allclose(ref_res, pd_res1)
        # test empty
        pd_res2 = paddle.cartesian_prod([a, b, c, d])
        ref_res = np.array(
            list(product(self.a_np, self.b_np, self.c_np, self.d_np))
        ).reshape([0, 4])
        np.testing.assert_allclose(ref_res, pd_res2)

        pd_res3 = paddle.cartesian_prod([a])
        ref_res = np.array(list(product(self.a_np)))
        np.testing.assert_allclose(ref_res.flatten(), pd_res3)


if __name__ == '__main__':
    unittest.main()
