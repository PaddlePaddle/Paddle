# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.base import core


class TestDLPack(unittest.TestCase):
    def test_dlpack_dygraph(self):
        with dygraph_guard():
            tensor = paddle.to_tensor(np.array([1, 2, 3, 4]).astype("int"))
            dlpack = paddle.utils.dlpack.to_dlpack(tensor)
            out_from_dlpack = paddle.utils.dlpack.from_dlpack(dlpack)
            self.assertTrue(
                isinstance(out_from_dlpack, paddle.base.core.eager.Tensor)
            )
            np.testing.assert_array_equal(
                out_from_dlpack.numpy(), np.array([1, 2, 3, 4]).astype("int")
            )

    def test_dlpack_tensor_larger_than_2dim(self):
        with dygraph_guard():
            numpy_data = np.random.randn(4, 5, 6)
            t = paddle.to_tensor(numpy_data)
            dlpack = paddle.utils.dlpack.to_dlpack(t)
            out = paddle.utils.dlpack.from_dlpack(dlpack)
            np.testing.assert_allclose(numpy_data, out.numpy(), rtol=1e-05)

    def test_dlpack_static(self):
        with static_guard():
            tensor = base.create_lod_tensor(
                np.array([[1], [2], [3], [4]]).astype("int"),
                [[1, 3]],
                base.CPUPlace(),
            )
            dlpack = paddle.utils.dlpack.to_dlpack(tensor)
            out_from_dlpack = paddle.utils.dlpack.from_dlpack(dlpack)
            self.assertTrue(isinstance(out_from_dlpack, base.core.Tensor))
            np.testing.assert_array_equal(
                np.array(out_from_dlpack),
                np.array([[1], [2], [3], [4]]).astype("int"),
            )

            # when build with cuda
            if core.is_compiled_with_cuda():
                gtensor = base.create_lod_tensor(
                    np.array([[1], [2], [3], [4]]).astype("int"),
                    [[1, 3]],
                    base.CUDAPlace(0),
                )
                gdlpack = paddle.utils.dlpack.to_dlpack(gtensor)
                gout_from_dlpack = paddle.utils.dlpack.from_dlpack(gdlpack)
                self.assertTrue(isinstance(gout_from_dlpack, base.core.Tensor))
                np.testing.assert_array_equal(
                    np.array(gout_from_dlpack),
                    np.array([[1], [2], [3], [4]]).astype("int"),
                )

    def test_dlpack_dtype_and_place_consistency(self):
        with dygraph_guard():
            dtypes = [
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "bool",
            ]
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
                places.append(base.CUDAPinnedPlace())
                dtypes.append("bfloat16")

            data = np.ones((2, 3, 4))
            for place in places:
                for dtype in dtypes:
                    x = paddle.to_tensor(data, dtype=dtype, place=place)
                    dlpack = paddle.utils.dlpack.to_dlpack(x)
                    o = paddle.utils.dlpack.from_dlpack(dlpack)
                    self.assertEqual(x.dtype, o.dtype)
                    np.testing.assert_allclose(x.numpy(), o.numpy(), rtol=1e-05)
                    self.assertEqual(type(x.place), type(o.place))

            complex_dtypes = ["complex64", "complex128"]
            for place in places:
                for dtype in complex_dtypes:
                    x = paddle.to_tensor(
                        [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]],
                        dtype=dtype,
                        place=place,
                    )
                    dlpack = paddle.utils.dlpack.to_dlpack(x)
                    o = paddle.utils.dlpack.from_dlpack(dlpack)
                    self.assertEqual(x.dtype, o.dtype)
                    np.testing.assert_allclose(x.numpy(), o.numpy(), rtol=1e-05)
                    self.assertEqual(type(x.place), type(o.place))

    def test_dlpack_deletion(self):
        # See Paddle issue 47171
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    a = paddle.rand(shape=[3, 5], dtype="float32").to(
                        device=place
                    )
                    dlpack = paddle.utils.dlpack.to_dlpack(a)
                    b = paddle.utils.dlpack.from_dlpack(dlpack)

    def test_to_dlpack_for_loop(self):
        # See Paddle issue 50120
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([3, 5]).to(device=place)
                    dlpack = paddle.utils.dlpack.to_dlpack(x)

    def test_to_dlpack_modification(self):
        # See Paddle issue 50120
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([3, 5]).to(device=place)
                    dlpack = paddle.utils.dlpack.to_dlpack(x)
                    y = paddle.utils.dlpack.from_dlpack(dlpack)
                    y[1:2, 2:5] = 2.0
                    np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_to_dlpack_data_ptr_consistency(self):
        # See Paddle issue 50120
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([3, 5]).to(device=place)
                    dlpack = paddle.utils.dlpack.to_dlpack(x)
                    y = paddle.utils.dlpack.from_dlpack(dlpack)

                    self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_to_dlpack_strides_consistency(self):
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([10, 10]).to(device=place)
                    x_strided = x[::2, ::2]
                    dlpack = paddle.utils.dlpack.to_dlpack(x_strided)
                    y = paddle.utils.dlpack.from_dlpack(dlpack)

                    self.assertEqual(x_strided.strides, y.strides)

    def test_to_dlpack_from_ext_tensor(self):
        with dygraph_guard():
            for _ in range(4):
                x = np.random.randn(3, 5)
                y = paddle.utils.dlpack.from_dlpack(x)

                self.assertEqual(x.__array_interface__['data'][0], y.data_ptr())
                np.testing.assert_allclose(x, y.numpy())


class TestRaiseError(unittest.TestCase):
    def test_to_dlpack_raise_type_error(self):
        self.assertRaises(TypeError, paddle.utils.dlpack.to_dlpack, np.zeros(5))


if __name__ == "__main__":
    unittest.main()
