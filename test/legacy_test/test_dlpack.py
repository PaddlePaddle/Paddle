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
            dlpack_v1 = paddle.utils.dlpack.to_dlpack(tensor)
            out_from_dlpack_v1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
            dlpack_v2 = paddle.to_dlpack(tensor)
            out_from_dlpack_v2 = paddle.from_dlpack(dlpack_v2)
            self.assertTrue(
                isinstance(out_from_dlpack_v1, paddle.base.core.eager.Tensor)
            )
            self.assertTrue(
                isinstance(out_from_dlpack_v2, paddle.base.core.eager.Tensor)
            )
            self.assertEqual(str(tensor.place), str(out_from_dlpack_v1.place))
            self.assertEqual(str(tensor.place), str(out_from_dlpack_v2.place))
            np.testing.assert_array_equal(
                out_from_dlpack_v1.numpy(), np.array([1, 2, 3, 4]).astype("int")
            )
            np.testing.assert_array_equal(
                out_from_dlpack_v2.numpy(), np.array([1, 2, 3, 4]).astype("int")
            )

    def test_dlpack_tensor_larger_than_2dim(self):
        with dygraph_guard():
            numpy_data = np.random.randn(4, 5, 6)
            t = paddle.to_tensor(numpy_data)
            dlpack_v1 = paddle.utils.dlpack.to_dlpack(t)
            dlpack_v2 = paddle.to_dlpack(t)
            out_v1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
            out_v2 = paddle.from_dlpack(dlpack_v2)
            self.assertEqual(str(t.place), str(out_v1.place))
            self.assertEqual(str(t.place), str(out_v2.place))
            np.testing.assert_allclose(numpy_data, out_v1.numpy(), rtol=1e-05)
            np.testing.assert_allclose(numpy_data, out_v2.numpy(), rtol=1e-05)

    def test_dlpack_static(self):
        with static_guard():
            tensor = base.create_lod_tensor(
                np.array([[1], [2], [3], [4]]).astype("int"),
                [[1, 3]],
                base.CPUPlace(),
            )
            dlpack_v1 = paddle.utils.dlpack.to_dlpack(tensor)
            out_from_dlpack_v1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
            dlpack_v2 = paddle.to_dlpack(tensor)
            out_from_dlpack_v2 = paddle.from_dlpack(dlpack_v2)
            self.assertTrue(
                isinstance(out_from_dlpack_v1, base.core.DenseTensor)
            )
            self.assertTrue(
                isinstance(out_from_dlpack_v2, base.core.DenseTensor)
            )
            np.testing.assert_array_equal(
                np.array(out_from_dlpack_v1),
                np.array([[1], [2], [3], [4]]).astype("int"),
            )
            np.testing.assert_array_equal(
                np.array(out_from_dlpack_v2),
                np.array([[1], [2], [3], [4]]).astype("int"),
            )

            # when build with cuda
            if core.is_compiled_with_cuda():
                gtensor = base.create_lod_tensor(
                    np.array([[1], [2], [3], [4]]).astype("int"),
                    [[1, 3]],
                    base.CUDAPlace(0),
                )
                gdlpack_v1 = paddle.utils.dlpack.to_dlpack(gtensor)
                gdlpack_v2 = paddle.to_dlpack(gtensor)
                gout_from_dlpack_v1 = paddle.utils.dlpack.from_dlpack(
                    gdlpack_v1
                )
                gout_from_dlpack_v2 = paddle.from_dlpack(gdlpack_v2)
                self.assertTrue(
                    isinstance(gout_from_dlpack_v1, base.core.DenseTensor)
                )
                self.assertTrue(
                    isinstance(gout_from_dlpack_v2, base.core.DenseTensor)
                )
                np.testing.assert_array_equal(
                    np.array(gout_from_dlpack_v1),
                    np.array([[1], [2], [3], [4]]).astype("int"),
                )
                np.testing.assert_array_equal(
                    np.array(gout_from_dlpack_v2),
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
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x)
                    o_v1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    dlpack_v2 = paddle.to_dlpack(x)
                    o_v2 = paddle.from_dlpack(dlpack_v2)
                    self.assertEqual(x.dtype, o_v1.dtype)
                    self.assertEqual(x.dtype, o_v2.dtype)
                    np.testing.assert_allclose(
                        x.numpy(), o_v1.numpy(), rtol=1e-05
                    )
                    np.testing.assert_allclose(
                        x.numpy(), o_v2.numpy(), rtol=1e-05
                    )
                    self.assertEqual(str(x.place), str(o_v1.place))
                    self.assertEqual(str(x.place), str(o_v2.place))

            complex_dtypes = ["complex64", "complex128"]
            for place in places:
                for dtype in complex_dtypes:
                    x = paddle.to_tensor(
                        [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]],
                        dtype=dtype,
                        place=place,
                    )
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x)
                    o_v1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    dlpack_v2 = paddle.to_dlpack(x)
                    o_v2 = paddle.from_dlpack(dlpack_v2)
                    self.assertEqual(x.dtype, o_v1.dtype)
                    self.assertEqual(x.dtype, o_v2.dtype)
                    np.testing.assert_allclose(
                        x.numpy(), o_v1.numpy(), rtol=1e-05
                    )
                    np.testing.assert_allclose(
                        x.numpy(), o_v2.numpy(), rtol=1e-05
                    )
                    self.assertEqual(str(x.place), str(o_v1.place))
                    self.assertEqual(str(x.place), str(o_v2.place))

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
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(a)
                    dlpack_v2 = paddle.to_dlpack(a)
                    b1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    b2 = paddle.from_dlpack(dlpack_v2)
                    self.assertEqual(str(a.place), str(b1.place))
                    self.assertEqual(str(a.place), str(b2.place))

    def test_to_dlpack_for_loop(self):
        # See Paddle issue 50120
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([3, 5]).to(device=place)
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x)
                    dlpack_v2 = paddle.to_dlpack(x)

    def test_to_dlpack_modification(self):
        # See Paddle issue 50120
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([3, 5]).to(device=place)
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x)
                    dlpack_v2 = paddle.to_dlpack(x)
                    y1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    y2 = paddle.from_dlpack(dlpack_v2)
                    y1[1:2, 2:5] = 2.0
                    y2[1:2, 2:5] = 2.0
                    np.testing.assert_allclose(x.numpy(), y1.numpy())
                    np.testing.assert_allclose(x.numpy(), y2.numpy())
                    self.assertEqual(str(x.place), str(y1.place))
                    self.assertEqual(str(x.place), str(y2.place))

    def test_to_dlpack_data_ptr_consistency(self):
        # See Paddle issue 50120
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([3, 5]).to(device=place)
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x)
                    dlpack_v2 = paddle.to_dlpack(x)
                    y1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    y2 = paddle.from_dlpack(dlpack_v2)

                    self.assertEqual(x.data_ptr(), y1.data_ptr())
                    self.assertEqual(x.data_ptr(), y2.data_ptr())
                    self.assertEqual(str(x.place), str(y1.place))
                    self.assertEqual(str(x.place), str(y2.place))

    def test_to_dlpack_strides_consistency(self):
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.rand([10, 10]).to(device=place)
                    x_strided = x[::2, ::2]
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x_strided)
                    dlpack_v2 = paddle.to_dlpack(x_strided)
                    y1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    y2 = paddle.from_dlpack(dlpack_v2)

                    self.assertEqual(x_strided.strides, y1.strides)
                    self.assertEqual(x_strided.strides, y2.strides)
                    self.assertEqual(str(x_strided.place), str(y1.place))
                    self.assertEqual(str(x_strided.place), str(y2.place))
                    np.testing.assert_equal(x_strided.numpy(), y1.numpy())
                    np.testing.assert_equal(x_strided.numpy(), y2.numpy())

    def test_to_dlpack_from_ext_tensor(self):
        with dygraph_guard():
            for _ in range(4):
                x = np.random.randn(3, 5)
                y1 = paddle.utils.dlpack.from_dlpack(x)
                y2 = paddle.from_dlpack(x)

                self.assertEqual(
                    x.__array_interface__['data'][0], y1.data_ptr()
                )
                self.assertEqual(
                    x.__array_interface__['data'][0], y2.data_ptr()
                )
                np.testing.assert_allclose(x, y1.numpy())
                np.testing.assert_allclose(x, y2.numpy())

    def test_to_dlpack_from_zero_dim(self):
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.to_tensor(1.0, place=place)
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x)
                    dlpack_v2 = paddle.to_dlpack(x)
                    y1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    y2 = paddle.from_dlpack(dlpack_v2)
                    self.assertEqual(x.data_ptr(), y1.data_ptr())
                    self.assertEqual(x.data_ptr(), y2.data_ptr())
                    self.assertEqual(str(x.place), str(y1.place))
                    self.assertEqual(str(x.place), str(y2.place))
                    self.assertEqual(y1.shape, [])
                    self.assertEqual(y2.shape, [])
                    self.assertEqual(y1.numel().item(), 1)
                    self.assertEqual(y2.numel().item(), 1)
                    np.testing.assert_array_equal(x.numpy(), y1.numpy())
                    np.testing.assert_array_equal(x.numpy(), y2.numpy())

    def test_to_dlpack_from_zero_size(self):
        with dygraph_guard():
            places = [base.CPUPlace()]
            if paddle.is_compiled_with_cuda():
                places.append(base.CUDAPlace(0))
            for place in places:
                for _ in range(4):
                    x = paddle.zeros([0, 10]).to(device=place)
                    dlpack_v1 = paddle.utils.dlpack.to_dlpack(x)
                    dlpack_v2 = paddle.to_dlpack(x)
                    y1 = paddle.utils.dlpack.from_dlpack(dlpack_v1)
                    y2 = paddle.from_dlpack(dlpack_v2)
                    self.assertEqual(x.data_ptr(), y1.data_ptr())
                    self.assertEqual(x.data_ptr(), y2.data_ptr())
                    self.assertEqual(str(x.place), str(y1.place))
                    self.assertEqual(str(x.place), str(y2.place))
                    self.assertEqual(y1.shape, [0, 10])
                    self.assertEqual(y2.shape, [0, 10])
                    self.assertEqual(y1.numel().item(), 0)
                    self.assertEqual(y2.numel().item(), 0)
                    np.testing.assert_array_equal(x.numpy(), y1.numpy())
                    np.testing.assert_array_equal(x.numpy(), y2.numpy())


from paddle.utils.dlpack import DLDeviceType


class TestDLPackDevice(unittest.TestCase):
    def test_dlpack_device(self):
        with dygraph_guard():

            tensor_cpu = paddle.to_tensor([1, 2, 3], place=base.CPUPlace())
            device_type, device_id = tensor_cpu.__dlpack_device__()
            self.assertEqual(device_type, DLDeviceType.kDLCPU)
            self.assertEqual(device_id, None)

            if paddle.is_compiled_with_cuda():
                tensor_cuda = paddle.to_tensor(
                    [1, 2, 3], place=base.CUDAPlace(0)
                )
                device_type, device_id = tensor_cuda.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDA)
                self.assertEqual(device_id, 0)

            if paddle.is_compiled_with_cuda():
                tensor_pinned = paddle.to_tensor(
                    [1, 2, 3], place=base.CUDAPinnedPlace()
                )
                device_type, device_id = tensor_pinned.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDAHost)
                self.assertEqual(device_id, None)

            if paddle.is_compiled_with_xpu():
                tensor_xpu = paddle.to_tensor([1, 2, 3], place=base.XPUPlace(0))
                device_type, device_id = tensor_xpu.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLOneAPI)
                self.assertEqual(device_id, 0)

    def test_dlpack_device_zero_dim(self):
        with dygraph_guard():

            tensor = paddle.to_tensor(5.0, place=base.CPUPlace())
            device_type, device_id = tensor.__dlpack_device__()
            self.assertEqual(device_type, DLDeviceType.kDLCPU)
            self.assertEqual(device_id, None)

            if paddle.is_compiled_with_cuda():
                tensor_cuda = paddle.to_tensor(5.0, place=base.CUDAPlace(0))
                device_type, device_id = tensor_cuda.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDA)
                self.assertEqual(device_id, 0)

            if paddle.is_compiled_with_xpu():
                tensor_xpu = paddle.to_tensor(5.0, place=base.XPUPlace(0))
                device_type, device_id = tensor_xpu.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLOneAPI)
                self.assertEqual(device_id, 0)

    def test_dlpack_device_zero_size(self):
        with dygraph_guard():
            tensor = paddle.to_tensor(
                paddle.zeros([0, 10]), place=base.CPUPlace()
            )
            device_type, device_id = tensor.__dlpack_device__()
            self.assertEqual(device_type, DLDeviceType.kDLCPU)
            self.assertEqual(device_id, None)

            if paddle.is_compiled_with_cuda():
                tensor_cuda = paddle.to_tensor(
                    paddle.zeros([0, 10]), place=base.CUDAPlace(0)
                )
                device_type, device_id = tensor_cuda.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDA)
                self.assertEqual(device_id, 0)

            if paddle.is_compiled_with_xpu():
                tensor_xpu = paddle.to_tensor(
                    paddle.zeros([0, 10]), place=base.XPUPlace(0)
                )
                device_type, device_id = tensor_xpu.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLOneAPI)
                self.assertEqual(device_id, 0)


class TestRaiseError(unittest.TestCase):
    def test_to_dlpack_raise_type_error(self):
        self.assertRaises(TypeError, paddle.utils.dlpack.to_dlpack, np.zeros(5))
        self.assertRaises(TypeError, paddle.to_dlpack, np.zeros(5))


if __name__ == "__main__":
    unittest.main()
