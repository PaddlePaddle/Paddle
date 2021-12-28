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

import paddle.fluid.core as core
import paddle.fluid.eager.eager_tensor_patch_methods as eager_tensor_patch_methods
import paddle
import numpy as np
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid.data_feeder import convert_dtype
import unittest


class EagerScaleTestCase(unittest.TestCase):
    def test_scale_base(self):
        with _test_eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            tensor = paddle.to_tensor(arr, 'float32', core.CPUPlace())
            print(tensor)
            tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            for i in range(0, 100):
                tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            print(tensor)
            self.assertEqual(tensor.shape, [4, 16, 16, 32])
            self.assertEqual(tensor.stop_gradient, True)

    def test_retain_grad_and_run_backward(self):
        with _test_eager_guard():
            paddle.set_device("cpu")

            input_data = np.ones([4, 16, 16, 32]).astype('float32')
            data_eager = paddle.to_tensor(input_data, 'float32',
                                          core.CPUPlace(), False)

            grad_data = np.ones([4, 16, 16, 32]).astype('float32')
            grad_eager = paddle.to_tensor(grad_data, 'float32', core.CPUPlace())

            core.eager.retain_grad_for_tensor(data_eager)

            out_eager = core.eager.scale(data_eager, 1.0, 0.9, True, True)
            self.assertFalse(data_eager.grad._is_initialized())
            core.eager.run_backward([out_eager], [grad_eager], False)
            self.assertTrue(data_eager.grad._is_initialized())
            self.assertTrue(np.array_equal(data_eager.grad.numpy(), input_data))


class EagerDtypeTestCase(unittest.TestCase):
    def check_to_tesnsor_and_numpy(self, dtype, proto_dtype):
        with _test_eager_guard():
            arr = np.random.random([4, 16, 16, 32]).astype(dtype)
            tensor = paddle.to_tensor(arr, dtype)
            self.assertEqual(tensor.dtype, proto_dtype)
            self.assertTrue(np.array_equal(arr, tensor.numpy()))

    def test_dtype_base(self):
        print("Test_dtype")
        self.check_to_tesnsor_and_numpy('bool', core.VarDesc.VarType.BOOL)
        self.check_to_tesnsor_and_numpy('int8', core.VarDesc.VarType.INT8)
        self.check_to_tesnsor_and_numpy('uint8', core.VarDesc.VarType.UINT8)
        self.check_to_tesnsor_and_numpy('int16', core.VarDesc.VarType.INT16)
        self.check_to_tesnsor_and_numpy('int32', core.VarDesc.VarType.INT32)
        self.check_to_tesnsor_and_numpy('int64', core.VarDesc.VarType.INT64)
        self.check_to_tesnsor_and_numpy('float16', core.VarDesc.VarType.FP16)
        self.check_to_tesnsor_and_numpy('float32', core.VarDesc.VarType.FP32)
        self.check_to_tesnsor_and_numpy('float64', core.VarDesc.VarType.FP64)
        self.check_to_tesnsor_and_numpy('complex64',
                                        core.VarDesc.VarType.COMPLEX64)
        self.check_to_tesnsor_and_numpy('complex128',
                                        core.VarDesc.VarType.COMPLEX128)


class EagerTensorPropertiesTestCase(unittest.TestCase):
    def constructor(self, place):
        egr_tensor = core.eager.EagerTensor()
        self.assertEqual(egr_tensor.persistable, False)
        self.assertTrue("generated" in egr_tensor.name)
        self.assertEqual(egr_tensor.shape, [])
        self.assertEqual(egr_tensor.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor.stop_gradient, True)

        egr_tensor0 = core.eager.EagerTensor(
            core.VarDesc.VarType.FP32, [4, 16, 16, 32], "test_eager_tensor",
            core.VarDesc.VarType.LOD_TENSOR, True)
        self.assertEqual(egr_tensor0.persistable, True)
        self.assertEqual(egr_tensor0.name, "test_eager_tensor")
        self.assertEqual(egr_tensor0.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor0.dtype, core.VarDesc.VarType.FP32)

        arr0 = np.random.rand(4, 16, 16, 32).astype('float32')
        egr_tensor1 = core.eager.EagerTensor(arr0, place, True, False,
                                             "numpy_tensor1", False)
        self.assertEqual(egr_tensor1.persistable, True)
        self.assertEqual(egr_tensor1.name, "numpy_tensor1")
        self.assertEqual(egr_tensor1.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor1.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor1.stop_gradient, False)
        self.assertTrue(egr_tensor1.place._equals(place))
        self.assertTrue(np.array_equal(egr_tensor1.numpy(), arr0))

        arr1 = np.random.randint(100, size=(4, 16, 16, 32), dtype=np.int64)
        egr_tensor2 = core.eager.EagerTensor(arr1, place, False, True,
                                             "numpy_tensor2", True)
        self.assertEqual(egr_tensor2.persistable, False)
        self.assertEqual(egr_tensor2.name, "numpy_tensor2")
        self.assertEqual(egr_tensor2.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor2.dtype, core.VarDesc.VarType.INT64)
        self.assertEqual(egr_tensor2.stop_gradient, True)
        self.assertTrue(egr_tensor2.place._equals(place))
        self.assertTrue(np.array_equal(egr_tensor2.numpy(), arr1))

        arr2 = np.random.rand(4, 16, 16, 32, 64).astype('float32')
        egr_tensor3 = core.eager.EagerTensor(arr2)
        self.assertEqual(egr_tensor3.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor3.name)
        self.assertEqual(egr_tensor3.shape, [4, 16, 16, 32, 64])
        self.assertEqual(egr_tensor3.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor3.stop_gradient, True)
        self.assertTrue(
            egr_tensor3.place._equals(
                paddle.fluid.framework._current_expected_place()))
        self.assertTrue(np.array_equal(egr_tensor3.numpy(), arr2))

        egr_tensor3.stop_gradient = False
        egr_tensor4 = core.eager.EagerTensor(egr_tensor3)
        self.assertEqual(egr_tensor4.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor4.name)
        self.assertEqual(egr_tensor4.shape, egr_tensor3.shape)
        self.assertEqual(egr_tensor4.dtype, egr_tensor3.dtype)
        self.assertEqual(egr_tensor4.stop_gradient, True)
        self.assertTrue(
            egr_tensor4.place._equals(
                paddle.fluid.framework._current_expected_place()))
        self.assertTrue(
            np.array_equal(egr_tensor4.numpy(), egr_tensor3.numpy()))

        arr4 = np.random.rand(4, 16, 16, 32).astype('float32')
        egr_tensor5 = core.eager.EagerTensor(arr4, place)
        self.assertEqual(egr_tensor5.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor5.name)
        self.assertEqual(egr_tensor5.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor5.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor5.stop_gradient, True)
        self.assertTrue(egr_tensor5.place._equals(place))
        self.assertTrue(np.array_equal(egr_tensor5.numpy(), arr4))

        egr_tensor6 = core.eager.EagerTensor(egr_tensor5, core.CPUPlace())
        self.assertEqual(egr_tensor6.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor6.name)
        self.assertEqual(egr_tensor6.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor6.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor6.stop_gradient, True)
        self.assertEqual(egr_tensor6.place.is_cpu_place(), True)
        self.assertTrue(
            np.array_equal(egr_tensor6.numpy(), egr_tensor5.numpy()))

        egr_tensor7 = core.eager.EagerTensor(arr4, place, True)
        self.assertEqual(egr_tensor7.persistable, True)
        self.assertTrue("generated_tensor" in egr_tensor7.name)
        self.assertEqual(egr_tensor7.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor7.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor7.stop_gradient, True)
        self.assertTrue(egr_tensor7.place._equals(place))
        self.assertTrue(np.array_equal(egr_tensor7.numpy(), arr4))

        egr_tensor8 = core.eager.EagerTensor(egr_tensor6, place, "egr_tensor8")
        self.assertEqual(egr_tensor8.persistable, False)
        self.assertEqual(egr_tensor8.name, "egr_tensor8")
        self.assertEqual(egr_tensor8.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor8.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor8.stop_gradient, True)
        self.assertTrue(egr_tensor8.place._equals(place))
        self.assertTrue(
            np.array_equal(egr_tensor8.numpy(), egr_tensor5.numpy()))

        egr_tensor9 = core.eager.EagerTensor(arr4, place, True, True)
        self.assertEqual(egr_tensor9.persistable, True)
        self.assertTrue("generated_tensor" in egr_tensor9.name)
        self.assertEqual(egr_tensor9.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor9.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor9.stop_gradient, True)
        self.assertTrue(egr_tensor9.place._equals(place))
        self.assertTrue(np.array_equal(egr_tensor9.numpy(), arr4))

    def test_constructor(self):
        print("Test_constructor")
        paddle.set_device("cpu")
        place_list = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            place_list.append(core.CUDAPlace(0))
        with _test_eager_guard():
            for p in place_list:
                self.constructor(p)

    def constructor_with_kwargs(self, place):
        # init EagerTensor by Python array
        arr = np.random.rand(4, 16, 16, 32).astype('float32')

        egr_tensor0 = core.eager.EagerTensor(value=arr)
        self.assertEqual(egr_tensor0.persistable, False)
        self.assertTrue("generated" in egr_tensor0.name)
        self.assertEqual(egr_tensor0.shape, [4, 16, 16, 32])
        self.assertTrue(
            egr_tensor0.place._equals(
                paddle.fluid.framework._current_expected_place()))
        self.assertEqual(egr_tensor0.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor0.stop_gradient, True)

        egr_tensor1 = core.eager.EagerTensor(value=arr, place=place)
        self.assertEqual(egr_tensor1.persistable, False)
        self.assertTrue("generated" in egr_tensor1.name)
        self.assertEqual(egr_tensor1.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor1.place._equals(place))
        self.assertEqual(egr_tensor1.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor1.stop_gradient, True)

        egr_tensor2 = core.eager.EagerTensor(arr, place=place)
        self.assertEqual(egr_tensor2.persistable, False)
        self.assertTrue("generated" in egr_tensor2.name)
        self.assertEqual(egr_tensor2.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor2.place._equals(place))
        self.assertEqual(egr_tensor2.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor2.stop_gradient, True)

        egr_tensor3 = core.eager.EagerTensor(
            arr, place=place, name="new_eager_tensor")
        self.assertEqual(egr_tensor3.persistable, False)
        self.assertTrue("new_eager_tensor" in egr_tensor3.name)
        self.assertEqual(egr_tensor3.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor3.place._equals(place))
        self.assertEqual(egr_tensor3.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor3.stop_gradient, True)

        egr_tensor4 = core.eager.EagerTensor(
            arr, place=place, persistable=True, name="new_eager_tensor")
        self.assertEqual(egr_tensor4.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor4.name)
        self.assertEqual(egr_tensor4.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor4.place._equals(place))
        self.assertEqual(egr_tensor4.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor4.stop_gradient, True)

        egr_tensor5 = core.eager.EagerTensor(
            arr,
            core.CPUPlace(),
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True)
        self.assertEqual(egr_tensor5.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor5.name)
        self.assertEqual(egr_tensor5.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor5.place.is_cpu_place())
        self.assertEqual(egr_tensor5.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor5.stop_gradient, True)

        egr_tensor6 = core.eager.EagerTensor(
            arr,
            place=core.CPUPlace(),
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True)
        self.assertEqual(egr_tensor6.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor6.name)
        self.assertEqual(egr_tensor6.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor6.place.is_cpu_place())
        self.assertEqual(egr_tensor6.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor6.stop_gradient, True)

        egr_tensor7 = core.eager.EagerTensor(
            arr,
            place=place,
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True)
        self.assertEqual(egr_tensor7.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor7.name)
        self.assertEqual(egr_tensor7.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor7.place._equals(place))
        self.assertEqual(egr_tensor7.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor7.stop_gradient, True)

        egr_tensor8 = core.eager.EagerTensor(
            arr,
            place=place,
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True,
            stop_gradient=False)
        self.assertEqual(egr_tensor8.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor8.name)
        self.assertEqual(egr_tensor8.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor8.place._equals(place))
        self.assertEqual(egr_tensor8.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor8.stop_gradient, False)

        egr_tensor9 = core.eager.EagerTensor(
            arr, place, True, True, "new_eager_tensor", stop_gradient=False)
        self.assertEqual(egr_tensor9.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor9.name)
        self.assertEqual(egr_tensor9.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor9.place._equals(place))
        self.assertEqual(egr_tensor9.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor9.stop_gradient, False)

        egr_tensor10 = core.eager.EagerTensor(
            arr,
            place,
            True,
            True,
            name="new_eager_tensor",
            stop_gradient=False)
        self.assertEqual(egr_tensor10.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor10.name)
        self.assertEqual(egr_tensor10.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor10.place._equals(place))
        self.assertEqual(egr_tensor10.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor10.stop_gradient, False)

        egr_tensor11 = core.eager.EagerTensor(
            arr,
            place,
            True,
            zero_copy=True,
            name="new_eager_tensor",
            stop_gradient=False)
        self.assertEqual(egr_tensor11.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor11.name)
        self.assertEqual(egr_tensor11.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor11.place._equals(place))
        self.assertEqual(egr_tensor11.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor11.stop_gradient, False)

        egr_tensor12 = core.eager.EagerTensor(
            arr,
            place,
            persistable=True,
            zero_copy=True,
            name="new_eager_tensor",
            stop_gradient=False)
        self.assertEqual(egr_tensor12.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor12.name)
        self.assertEqual(egr_tensor12.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor12.place._equals(place))
        self.assertEqual(egr_tensor12.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor12.stop_gradient, False)

        egr_tensor13 = core.eager.EagerTensor(
            value=arr,
            place=place,
            persistable=True,
            zero_copy=True,
            name="new_eager_tensor",
            stop_gradient=False)
        self.assertEqual(egr_tensor13.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor13.name)
        self.assertEqual(egr_tensor13.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor13.place._equals(place))
        self.assertEqual(egr_tensor13.dtype, core.VarDesc.VarType.FP32)
        self.assertEqual(egr_tensor13.stop_gradient, False)

        # special case
        egr_tensor14 = core.eager.EagerTensor(
            dtype=core.VarDesc.VarType.FP32,
            dims=[4, 16, 16, 32],
            name="special_eager_tensor",
            vtype=core.VarDesc.VarType.LOD_TENSOR,
            persistable=True)
        self.assertEqual(egr_tensor14.persistable, True)
        self.assertEqual(egr_tensor14.name, "special_eager_tensor")
        self.assertEqual(egr_tensor14.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor14.dtype, core.VarDesc.VarType.FP32)

        # init EagerTensor by EagerTensor
        egr_tensor15 = core.eager.EagerTensor(value=egr_tensor4)
        self.assertEqual(egr_tensor15.persistable, True)
        self.assertTrue("generated" in egr_tensor15.name)
        self.assertEqual(egr_tensor15.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor15.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor15.stop_gradient, True)
        self.assertTrue(
            egr_tensor15.place._equals(
                paddle.fluid.framework._current_expected_place()))
        self.assertTrue(
            np.array_equal(egr_tensor15.numpy(), egr_tensor4.numpy()))

        egr_tensor16 = core.eager.EagerTensor(
            value=egr_tensor4, name="new_eager_tensor")
        self.assertEqual(egr_tensor16.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor16.name)
        self.assertEqual(egr_tensor16.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor16.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor16.stop_gradient, True)
        self.assertTrue(
            egr_tensor16.place._equals(
                paddle.fluid.framework._current_expected_place()))
        self.assertTrue(
            np.array_equal(egr_tensor16.numpy(), egr_tensor4.numpy()))

        egr_tensor17 = core.eager.EagerTensor(
            value=egr_tensor4,
            place=place,
            name="new_eager_tensor", )
        self.assertEqual(egr_tensor17.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor17.name)
        self.assertEqual(egr_tensor17.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor17.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor17.stop_gradient, True)
        self.assertTrue(egr_tensor17.place._equals(place))
        self.assertTrue(
            np.array_equal(egr_tensor17.numpy(), egr_tensor4.numpy()))

        egr_tensor18 = core.eager.EagerTensor(
            egr_tensor4,
            place=place,
            name="new_eager_tensor", )
        self.assertEqual(egr_tensor18.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor18.name)
        self.assertEqual(egr_tensor18.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor18.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor18.stop_gradient, True)
        self.assertTrue(egr_tensor18.place._equals(place))
        self.assertTrue(
            np.array_equal(egr_tensor18.numpy(), egr_tensor4.numpy()))

        egr_tensor19 = core.eager.EagerTensor(
            egr_tensor4,
            place,
            name="new_eager_tensor", )
        self.assertEqual(egr_tensor19.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor19.name)
        self.assertEqual(egr_tensor19.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor19.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor19.stop_gradient, True)
        self.assertTrue(egr_tensor19.place._equals(place))
        self.assertTrue(
            np.array_equal(egr_tensor19.numpy(), egr_tensor4.numpy()))

    def test_constructor_with_kwargs(self):
        print("Test_constructor_with_kwargs")
        paddle.set_device("cpu")
        place_list = [core.CPUPlace()]
        # if core.is_compiled_with_cuda():
        #     place_list.append(core.CUDAPlace(0))
        with _test_eager_guard():
            for p in place_list:
                self.constructor_with_kwargs(p)

    def test_copy_and_copy_to(self):
        print("Test_copy_and_copy_to")
        with _test_eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            arr1 = np.zeros([4, 16]).astype('float32')
            arr2 = np.ones([4, 16, 16, 32]).astype('float32') + np.ones(
                [4, 16, 16, 32]).astype('float32')
            tensor = paddle.to_tensor(arr, core.VarDesc.VarType.FP32,
                                      core.CPUPlace())
            self.assertEqual(tensor.stop_gradient, True)
            tensor.stop_gradient = False
            print("Set persistable")
            tensor.persistable = False
            tensor1 = paddle.to_tensor(arr1, core.VarDesc.VarType.FP32,
                                       core.CPUPlace())
            tensor1.persistable = True
            self.assertEqual(tensor1.stop_gradient, True)
            self.assertTrue(np.array_equal(tensor.numpy(), arr))
            print("Test copy_")
            tensor.copy_(tensor1, True)
            self.assertEqual(tensor.persistable, True)
            self.assertEqual(tensor.shape, [4, 16])
            self.assertEqual(tensor.dtype, core.VarDesc.VarType.FP32)
            self.assertTrue(np.array_equal(tensor.numpy(), arr1))

            print("Test _copy_to")
            tensor2 = paddle.to_tensor(arr2, core.VarDesc.VarType.FP32,
                                       core.CPUPlace())
            self.assertTrue(np.array_equal(tensor2.numpy(), arr2))
            self.assertTrue(tensor2.place.is_cpu_place())
            tensor2.persistable = True
            tensor2.stop_gradient = False
            if core.is_compiled_with_cuda():
                tensor3 = tensor2._copy_to(True, core.CUDAPlace(0))
                self.assertTrue(np.array_equal(tensor3.numpy(), arr2))
                self.assertTrue(tensor3.persistable, True)
                self.assertTrue(tensor3.stop_gradient, True)
                self.assertTrue(tensor3.place.is_gpu_place())
            else:
                tensor3 = tensor2._copy_to(True, core.CPUPlace())
                self.assertTrue(np.array_equal(tensor3.numpy(), arr2))
                self.assertTrue(tensor3.persistable, True)
                self.assertTrue(tensor3.stop_gradient, True)
                self.assertTrue(tensor3.place.is_cpu_place())

    def test_properties(self):
        print("Test_properties")
        with _test_eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            tensor = paddle.to_tensor(arr, core.VarDesc.VarType.FP32,
                                      core.CPUPlace())
            self.assertEqual(tensor.shape, [4, 16, 16, 32])
            tensor.name = 'tensor_name_test'
            self.assertEqual(tensor.name, 'tensor_name_test')
            self.assertEqual(tensor.persistable, False)
            tensor.persistable = True
            self.assertEqual(tensor.persistable, True)
            tensor.persistable = False
            self.assertEqual(tensor.persistable, False)
            self.assertTrue(tensor.place.is_cpu_place())
            self.assertEqual(tensor._place_str, 'CPUPlace')
            self.assertEqual(tensor.stop_gradient, True)
            tensor.stop_gradient = False
            self.assertEqual(tensor.stop_gradient, False)
            tensor.stop_gradient = True
            self.assertEqual(tensor.stop_gradient, True)

    def test_global_properties(self):
        print("Test_global_properties")
        self.assertFalse(core._in_eager_mode())
        with _test_eager_guard():
            self.assertTrue(core._in_eager_mode())
        self.assertFalse(core._in_eager_mode())

    def test_place_guard(self):
        core._enable_eager_mode()
        if core.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
            with paddle.fluid.framework._dygraph_place_guard(core.CPUPlace()):
                self.assertTrue(core.eager._get_expected_place().is_cpu_place())
        else:
            paddle.set_device("cpu")
            with paddle.fluid.framework._dygraph_place_guard(core.CPUPlace()):
                self.assertTrue(core.eager._get_expected_place().is_cpu_place())
        core._disable_eager_mode()


if __name__ == "__main__":
    unittest.main()
