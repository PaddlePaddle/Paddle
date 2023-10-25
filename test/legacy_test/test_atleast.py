# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class AtleastBase(unittest.TestCase):
    def setUp(self):
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def static_api(self, func):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                name='x', shape=self.x_shape, dtype=self.x_dtype
            )
            f = getattr(paddle, func)
            out = f(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x,
                },
                fetch_list=[out],
            )
            f = getattr(np, func)
            expect_output = f(self.x)
            np.testing.assert_allclose(expect_output, res[0], atol=1e-05)

    def dygraph_api(self, func):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        f = getattr(paddle, func)
        res = f(x)
        f = getattr(np, func)
        expect_output = f(self.x)

        np.testing.assert_allclose(expect_output, res.numpy(), atol=1e-05)
        paddle.enable_static()


class AtleastList(unittest.TestCase):
    def setUp(self):
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.y = np.array(0.5, dtype="float64")
        self.y_shape = []
        self.y_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        # numpy's atleast_*d not use to pytorch, for list, the arg , [np.array(1.0, dtype = "float64"), np.array(1.0, dtype = "float64")] , is 2 dimension array ,not a list of tensor
        # this expect output is the result for torch.atleast_*d((torch.tensor(1.), torch.tensor(0.5)))
        self.expect_output = np.array([[1.0], [0.5]])

    def static_api(self, func):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(
                name='x', shape=self.x_shape, dtype=self.x_dtype
            )
            y = paddle.static.data(
                name='y', shape=self.y_shape, dtype=self.y_dtype
            )
            f = getattr(paddle, func)
            out = f((x, y))
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[out],
            )

            np.testing.assert_allclose(self.expect_output, res, atol=1e-05)

    def dygraph_api(self, func):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        f = getattr(paddle, func)
        res = f((x, y))
        f = getattr(np, func)

        res_num = []
        for item in res:
            res_num += [item.numpy()]
        np.testing.assert_allclose(self.expect_output, res_num, atol=1e-05)
        paddle.enable_static()


class TestAtleast1D0D(AtleastBase):
    # test atleast_1d function using 0 dim tensor
    def setUp(self):
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_1d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_1d')


class TestAtleast1D1D(AtleastBase):
    # test atleast_1d function using 1 dim tensor
    def setUp(self):
        self.x = np.array([1.0], dtype="float64")
        self.x_shape = [1]
        self.x_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_1d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_1d')


class TestAtleast1D2D(AtleastBase):
    # test atleast_1d function using 2 dim tensor
    def setUp(self):
        self.x = np.arange(2, dtype="int64")
        self.x_shape = [2]
        self.x_dtype = "int64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_1d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_1d')


class TestAtleast1Dlist(AtleastList):
    # test atleast_1d function using list of tensor

    def test_static_api(self):
        super().static_api('atleast_1d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_1d')


class TestAtleast2D0D(AtleastBase):
    # test atleast_2d function using 0 dim tensor
    def setUp(self):
        # for 0 dim
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_2d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_2d')


class TestAtleast2D2D(AtleastBase):
    # test atleast_2d function using 2 dim tensor
    def setUp(self):
        # for 2 dim
        self.x = np.arange(2, dtype="int64")
        self.x_shape = [2]
        self.x_dtype = "int64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_2d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_2d')


class TestAtleast2Dlist(AtleastList):
    # test atleast_2d function using list of tensor
    def setUp(self):
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.y = np.array(0.5, dtype="float64")
        self.y_shape = []
        self.y_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        # numpy's atleast_*d not use to pytorch, for list, the arg , [np.array(1.0, dtype = "float64"), np.array(1.0, dtype = "float64")] , is 2 dimension array ,not a list of tensor
        # this expect output is the result for torch.atleast_2d((torch.tensor(1.), torch.tensor(0.5)))
        self.expect_output = np.array([[[1.0]], [[0.5]]])

    def test_static_api(self):
        super().static_api('atleast_2d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_2d')


class TestAtleast3D0D(AtleastBase):
    # test atleast_3d function using 0 dim tensor
    def setUp(self):
        # for 0 dim
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_3d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_3d')


class TestAtleast3D1D(AtleastBase):
    # test atleast_3d function using 0 dim tensor
    def setUp(self):
        # for 0 dim
        self.x = np.array([1.0], dtype="float64")
        self.x_shape = [1]
        self.x_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_3d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_3d')


class TestAtleast3D2D(AtleastBase):
    # test atleast_3d function using 2 dim tensor
    def setUp(self):
        self.x = np.arange(4, dtype="int64").reshape([2, 2])
        self.x_shape = [2, 2]
        self.x_dtype = "int64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_3d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_3d')


class TestAtleast3D3D(AtleastBase):
    # test atleast_3d function using 3 dim tensor
    def setUp(self):
        self.x = np.array([[[1]]], dtype="int64")
        self.x_shape = [1, 1, 1]
        self.x_dtype = "int64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        super().static_api('atleast_3d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_3d')


class TestAtleast3Dlist(AtleastList):
    # test atleast_3d function using list of tensor
    def setUp(self):
        self.x = np.array(1.0, dtype="float64")
        self.x_shape = []
        self.x_dtype = "float64"
        self.y = np.array(0.5, dtype="float64")
        self.y_shape = []
        self.y_dtype = "float64"
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        # numpy's atleast_*d not use to pytorch, for list, the arg , [np.array(1.0, dtype = "float64"), np.array(1.0, dtype = "float64")] , is 2 dimension array ,not a list of tensor
        # this expect output is the result for torch.atleast_3d((torch.tensor(1.), torch.tensor(0.5)))
        self.expect_output = np.array([[[[1.0]]], [[[0.5]]]])

    def test_static_api(self):
        super().static_api('atleast_3d')

    def test_dygraph_api(self):
        super().dygraph_api('atleast_3d')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
