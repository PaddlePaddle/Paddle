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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestDLPack(unittest.TestCase):
    def test_dlpack_dygraph(self):
        paddle.disable_static()
        tensor = paddle.to_tensor(np.array([1, 2, 3, 4]).astype('int'))
        dlpack = paddle.utils.dlpack.to_dlpack(tensor)
        out_from_dlpack = paddle.utils.dlpack.from_dlpack(dlpack)
        self.assertTrue(isinstance(out_from_dlpack, paddle.Tensor))
        self.assertTrue(
            np.array_equal(
                np.array(out_from_dlpack), np.array([1, 2, 3, 4]).astype(
                    'int')))

    def test_dlpack_tensor_larger_than_2dim(self):
        paddle.disable_static()
        numpy_data = np.random.randn(4, 5, 6)
        t = paddle.to_tensor(numpy_data)
        # TODO: There may be a reference count problem of to_dlpack.
        dlpack = paddle.utils.dlpack.to_dlpack(t)
        out = paddle.utils.dlpack.from_dlpack(dlpack)
        self.assertTrue(np.allclose(numpy_data, out.numpy()))

    def test_dlpack_static(self):
        paddle.enable_static()
        tensor = fluid.create_lod_tensor(
            np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]],
            fluid.CPUPlace())
        dlpack = paddle.utils.dlpack.to_dlpack(tensor)
        out_from_dlpack = paddle.utils.dlpack.from_dlpack(dlpack)
        self.assertTrue(isinstance(out_from_dlpack, fluid.core.Tensor))
        self.assertTrue(
            np.array_equal(
                np.array(out_from_dlpack),
                np.array([[1], [2], [3], [4]]).astype('int')))

        # when build with cuda
        if core.is_compiled_with_cuda():
            gtensor = fluid.create_lod_tensor(
                np.array([[1], [2], [3], [4]]).astype('int'), [[1, 3]],
                fluid.CUDAPlace(0))
            gdlpack = paddle.utils.dlpack.to_dlpack(gtensor)
            gout_from_dlpack = paddle.utils.dlpack.from_dlpack(gdlpack)
            self.assertTrue(isinstance(gout_from_dlpack, fluid.core.Tensor))
            self.assertTrue(
                np.array_equal(
                    np.array(gout_from_dlpack),
                    np.array([[1], [2], [3], [4]]).astype('int')))

    def test_dlpack_dtype_conversion(self):
        paddle.disable_static()
        # DLpack does not explicitly support bool data type.
        dtypes = [
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
        ]
        data = np.ones((2, 3, 4))
        for dtype in dtypes:
            x = paddle.to_tensor(data, dtype=dtype)
            dlpack = paddle.utils.dlpack.to_dlpack(x)
            o = paddle.utils.dlpack.from_dlpack(dlpack)
            self.assertEqual(x.dtype, o.dtype)
            self.assertTrue(np.allclose(x.numpy(), o.numpy()))

        complex_dtypes = ["complex64", "complex128"]
        for dtype in complex_dtypes:
            x = paddle.to_tensor(
                [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]],
                dtype=dtype)
            dlpack = paddle.utils.dlpack.to_dlpack(x)
            o = paddle.utils.dlpack.from_dlpack(dlpack)
            self.assertEqual(x.dtype, o.dtype)
            self.assertTrue(np.allclose(x.numpy(), o.numpy()))


class TestRaiseError(unittest.TestCase):
    def test_from_dlpack_raise_type_error(self):
        self.assertRaises(TypeError, paddle.utils.dlpack.from_dlpack,
                          np.zeros(5))

    def test_to_dlpack_raise_type_error(self):
        self.assertRaises(TypeError, paddle.utils.dlpack.to_dlpack, np.zeros(5))


if __name__ == '__main__':
    unittest.main()
