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

import copy
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.base.framework import (
    EagerParamBase,
    _current_expected_place,
    in_dygraph_mode,
)


class EagerScaleTestCase(unittest.TestCase):
    def test_scale_base(self):
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
        paddle.set_device("cpu")

        input_data = np.ones([4, 16, 16, 32]).astype('float32')
        data_eager = paddle.to_tensor(
            input_data, 'float32', core.CPUPlace(), False
        )

        grad_data = np.ones([4, 16, 16, 32]).astype('float32')
        grad_eager = paddle.to_tensor(grad_data, 'float32', core.CPUPlace())

        data_eager.retain_grads()

        out_eager = core.eager.scale(data_eager, 1.0, 0.9, True, True)
        self.assertIsNone(data_eager.grad)
        out_eager.backward(grad_eager, False)
        self.assertIsNotNone(data_eager.grad)
        np.testing.assert_array_equal(data_eager.grad.numpy(), input_data)

    def test_retain_grad_and_run_backward_raises(self):
        paddle.set_device("cpu")

        input_data = np.ones([4, 16, 16, 32]).astype('float32')
        data_eager = paddle.to_tensor(
            input_data, 'float32', core.CPUPlace(), False
        )

        grad_data = np.ones([4, 16, 16, 32]).astype('float32')
        grad_data2 = np.ones([4, 16]).astype('float32')
        grad_eager = paddle.to_tensor(grad_data, 'float32', core.CPUPlace())
        grad_eager2 = paddle.to_tensor(grad_data2, 'float32', core.CPUPlace())

        data_eager.retain_grads()

        out_eager = core.eager.scale(data_eager, 1.0, 0.9, True, True)
        self.assertIsNone(data_eager.grad)
        with self.assertRaisesRegex(
            AssertionError, "The type of grad_tensor must be paddle.Tensor"
        ):
            out_eager.backward(grad_data, False)

        with self.assertRaisesRegex(
            AssertionError,
            "Tensor shape not match, Tensor of grad_tensor /*",
        ):
            out_eager.backward(grad_eager2, False)


class EagerDtypeTestCase(unittest.TestCase):
    def check_to_tensor_and_numpy(self, dtype, paddle_dtype):
        arr = np.random.random([4, 16, 16, 32]).astype(dtype)
        tensor = paddle.to_tensor(arr, dtype)
        self.assertEqual(tensor.dtype, paddle_dtype)
        np.testing.assert_array_equal(arr, tensor.numpy())

    def test_dtype_base(self):
        print("Test_dtype")
        self.check_to_tensor_and_numpy('bool', paddle.bool)
        self.check_to_tensor_and_numpy('int8', paddle.int8)
        self.check_to_tensor_and_numpy('uint8', paddle.uint8)
        self.check_to_tensor_and_numpy('int16', paddle.int16)
        self.check_to_tensor_and_numpy('int32', paddle.int32)
        self.check_to_tensor_and_numpy('int64', paddle.int64)
        self.check_to_tensor_and_numpy('float16', paddle.float16)
        self.check_to_tensor_and_numpy('float32', paddle.float32)
        self.check_to_tensor_and_numpy('float64', paddle.float64)
        self.check_to_tensor_and_numpy('complex64', paddle.complex64)
        self.check_to_tensor_and_numpy('complex128', paddle.complex128)


class EagerVariablePropertiesAndMethodsTestCase(unittest.TestCase):
    def constructor(self, place):
        egr_tensor = core.eager.Tensor()
        self.assertEqual(egr_tensor.persistable, False)
        self.assertTrue("generated" in egr_tensor.name)
        self.assertEqual(egr_tensor.shape, [0])
        self.assertEqual(egr_tensor.dtype, paddle.float32)
        self.assertEqual(egr_tensor.stop_gradient, True)

        egr_tensor0 = core.eager.Tensor(
            core.VarDesc.VarType.FP32,
            [4, 16, 16, 32],
            "test_eager_tensor",
            core.VarDesc.VarType.DENSE_TENSOR,
            True,
        )
        self.assertEqual(egr_tensor0.persistable, True)
        self.assertEqual(egr_tensor0.name, "test_eager_tensor")
        self.assertEqual(egr_tensor0.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor0.dtype, paddle.float32)

        arr0 = np.random.rand(4, 16, 16, 32).astype('float32')
        egr_tensor1 = core.eager.Tensor(
            arr0, place, True, False, "numpy_tensor1", False
        )
        self.assertEqual(egr_tensor1.persistable, True)
        self.assertEqual(egr_tensor1.name, "numpy_tensor1")
        self.assertEqual(egr_tensor1.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor1.dtype, paddle.float32)
        self.assertEqual(egr_tensor1.stop_gradient, False)
        self.assertTrue(egr_tensor1.place._equals(place))
        np.testing.assert_array_equal(egr_tensor1.numpy(), arr0)

        arr1 = np.random.randint(100, size=(4, 16, 16, 32), dtype=np.int64)
        egr_tensor2 = core.eager.Tensor(
            arr1, place, False, True, "numpy_tensor2", True
        )
        self.assertEqual(egr_tensor2.persistable, False)
        self.assertEqual(egr_tensor2.name, "numpy_tensor2")
        self.assertEqual(egr_tensor2.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor2.dtype, paddle.int64)
        self.assertEqual(egr_tensor2.stop_gradient, True)
        self.assertTrue(egr_tensor2.place._equals(place))
        np.testing.assert_array_equal(egr_tensor2.numpy(), arr1)

        arr2 = np.random.rand(4, 16, 16, 32, 64).astype('float32')
        egr_tensor3 = core.eager.Tensor(arr2)
        self.assertEqual(egr_tensor3.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor3.name)
        self.assertEqual(egr_tensor3.shape, [4, 16, 16, 32, 64])
        self.assertEqual(egr_tensor3.dtype, paddle.float32)
        self.assertEqual(egr_tensor3.stop_gradient, True)
        self.assertTrue(
            egr_tensor3.place._equals(
                paddle.base.framework._current_expected_place()
            )
        )
        np.testing.assert_array_equal(egr_tensor3.numpy(), arr2)

        egr_tensor3.stop_gradient = False
        egr_tensor4 = core.eager.Tensor(egr_tensor3)
        self.assertEqual(egr_tensor4.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor4.name)
        self.assertEqual(egr_tensor4.shape, egr_tensor3.shape)
        self.assertEqual(egr_tensor4.dtype, egr_tensor3.dtype)
        self.assertEqual(egr_tensor4.stop_gradient, True)
        self.assertTrue(
            egr_tensor4.place._equals(
                paddle.base.framework._current_expected_place()
            )
        )
        np.testing.assert_array_equal(egr_tensor4.numpy(), egr_tensor3.numpy())

        arr4 = np.random.rand(4, 16, 16, 32).astype('float32')
        egr_tensor5 = core.eager.Tensor(arr4, place)
        self.assertEqual(egr_tensor5.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor5.name)
        self.assertEqual(egr_tensor5.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor5.dtype, paddle.float32)
        self.assertEqual(egr_tensor5.stop_gradient, True)
        self.assertTrue(egr_tensor5.place._equals(place))
        np.testing.assert_array_equal(egr_tensor5.numpy(), arr4)

        egr_tensor6 = core.eager.Tensor(egr_tensor5, core.CPUPlace())
        self.assertEqual(egr_tensor6.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor6.name)
        self.assertEqual(egr_tensor6.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor6.dtype, paddle.float32)
        self.assertEqual(egr_tensor6.stop_gradient, True)
        self.assertEqual(egr_tensor6.place.is_cpu_place(), True)
        np.testing.assert_array_equal(egr_tensor6.numpy(), egr_tensor5.numpy())

        egr_tensor7 = core.eager.Tensor(arr4, place, True)
        self.assertEqual(egr_tensor7.persistable, True)
        self.assertTrue("generated_tensor" in egr_tensor7.name)
        self.assertEqual(egr_tensor7.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor7.dtype, paddle.float32)
        self.assertEqual(egr_tensor7.stop_gradient, True)
        self.assertTrue(egr_tensor7.place._equals(place))
        np.testing.assert_array_equal(egr_tensor7.numpy(), arr4)

        egr_tensor8 = core.eager.Tensor(egr_tensor6, place, "egr_tensor8")
        self.assertEqual(egr_tensor8.persistable, False)
        self.assertEqual(egr_tensor8.name, "egr_tensor8")
        self.assertEqual(egr_tensor8.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor8.dtype, paddle.float32)
        self.assertEqual(egr_tensor8.stop_gradient, True)
        self.assertTrue(egr_tensor8.place._equals(place))
        np.testing.assert_array_equal(egr_tensor8.numpy(), egr_tensor5.numpy())

        egr_tensor9 = core.eager.Tensor(arr4, place, True, True)
        self.assertEqual(egr_tensor9.persistable, True)
        self.assertTrue("generated_tensor" in egr_tensor9.name)
        self.assertEqual(egr_tensor9.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor9.dtype, paddle.float32)
        self.assertEqual(egr_tensor9.stop_gradient, True)
        self.assertTrue(egr_tensor9.place._equals(place))
        np.testing.assert_array_equal(egr_tensor9.numpy(), arr4)

        x = np.random.rand(3, 3).astype('float32')
        t = paddle.base.Tensor()
        t.set(x, paddle.base.CPUPlace())
        egr_tensor10 = core.eager.Tensor(t, place)
        self.assertEqual(egr_tensor10.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor10.name)
        self.assertEqual(egr_tensor10.shape, [3, 3])
        self.assertEqual(egr_tensor10.dtype, paddle.float32)
        self.assertEqual(egr_tensor10.stop_gradient, True)
        self.assertTrue(egr_tensor10.place._equals(place))
        np.testing.assert_array_equal(egr_tensor10.numpy(), x)

        egr_tensor11 = core.eager.Tensor(t, place, "framework_constructed")
        self.assertEqual(egr_tensor11.persistable, False)
        self.assertTrue("framework_constructed" in egr_tensor11.name)
        self.assertEqual(egr_tensor11.shape, [3, 3])
        self.assertEqual(egr_tensor11.dtype, paddle.float32)
        self.assertEqual(egr_tensor11.stop_gradient, True)
        self.assertTrue(egr_tensor11.place._equals(place))
        np.testing.assert_array_equal(egr_tensor11.numpy(), x)

        egr_tensor12 = core.eager.Tensor(t)
        self.assertEqual(egr_tensor12.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor12.name)
        self.assertEqual(egr_tensor12.shape, [3, 3])
        self.assertEqual(egr_tensor12.dtype, paddle.float32)
        self.assertEqual(egr_tensor12.stop_gradient, True)
        self.assertTrue(egr_tensor12.place._equals(paddle.base.CPUPlace()))
        np.testing.assert_array_equal(egr_tensor12.numpy(), x)

        zero_dim_param = EagerParamBase(shape=[], dtype="float32")
        self.assertEqual(zero_dim_param.shape, [])

        with self.assertRaisesRegex(
            ValueError, "The shape of Parameter should not be None"
        ):
            eager_param = EagerParamBase(shape=None, dtype="float32")

        with self.assertRaisesRegex(
            ValueError, "The dtype of Parameter should not be None"
        ):
            eager_param = EagerParamBase(shape=[1, 1], dtype=None)

        with self.assertRaisesRegex(
            ValueError,
            "Each dimension of shape for Parameter must be greater than 0, but received /*",
        ):
            eager_param = EagerParamBase(shape=[-1], dtype="float32")

        eager_param = EagerParamBase(shape=[1, 1], dtype="float32")
        self.assertTrue(eager_param.trainable)
        eager_param.trainable = False
        self.assertFalse(eager_param.trainable)
        with self.assertRaisesRegex(
            ValueError, "The type of trainable MUST be bool, but the type is /*"
        ):
            eager_param.trainable = "False"

        eager_param_2 = EagerParamBase(
            shape=paddle.shape(paddle.to_tensor([1, 2, 3, 4])), dtype="float32"
        )
        self.assertTrue(eager_param_2.trainable)
        eager_param_2.trainable = False
        self.assertFalse(eager_param_2.trainable)
        with self.assertRaisesRegex(
            ValueError, "The type of trainable MUST be bool, but the type is /*"
        ):
            eager_param_2.trainable = "False"

    def test_constructor(self):
        print("Test_constructor")
        paddle.set_device("cpu")
        place_list = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            place_list.append(core.CUDAPlace(0))

        for p in place_list:
            self.constructor(p)

    def constructor_with_kwargs(self, place):
        # init Tensor by Python array
        arr = np.random.rand(4, 16, 16, 32).astype('float32')

        egr_tensor0 = core.eager.Tensor(value=arr)
        self.assertEqual(egr_tensor0.persistable, False)
        self.assertTrue("generated" in egr_tensor0.name)
        self.assertEqual(egr_tensor0.shape, [4, 16, 16, 32])
        self.assertTrue(
            egr_tensor0.place._equals(
                paddle.base.framework._current_expected_place()
            )
        )
        self.assertEqual(egr_tensor0.dtype, paddle.float32)
        self.assertEqual(egr_tensor0.stop_gradient, True)

        egr_tensor1 = core.eager.Tensor(value=arr, place=place)
        self.assertEqual(egr_tensor1.persistable, False)
        self.assertTrue("generated" in egr_tensor1.name)
        self.assertEqual(egr_tensor1.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor1.place._equals(place))
        self.assertEqual(egr_tensor1.dtype, paddle.float32)
        self.assertEqual(egr_tensor1.stop_gradient, True)

        egr_tensor2 = core.eager.Tensor(arr, place=place)
        self.assertEqual(egr_tensor2.persistable, False)
        self.assertTrue("generated" in egr_tensor2.name)
        self.assertEqual(egr_tensor2.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor2.place._equals(place))
        self.assertEqual(egr_tensor2.dtype, paddle.float32)
        self.assertEqual(egr_tensor2.stop_gradient, True)

        egr_tensor3 = core.eager.Tensor(
            arr, place=place, name="new_eager_tensor"
        )
        self.assertEqual(egr_tensor3.persistable, False)
        self.assertTrue("new_eager_tensor" in egr_tensor3.name)
        self.assertEqual(egr_tensor3.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor3.place._equals(place))
        self.assertEqual(egr_tensor3.dtype, paddle.float32)
        self.assertEqual(egr_tensor3.stop_gradient, True)

        egr_tensor4 = core.eager.Tensor(
            arr, place=place, persistable=True, name="new_eager_tensor"
        )
        self.assertEqual(egr_tensor4.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor4.name)
        self.assertEqual(egr_tensor4.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor4.place._equals(place))
        self.assertEqual(egr_tensor4.dtype, paddle.float32)
        self.assertEqual(egr_tensor4.stop_gradient, True)

        egr_tensor5 = core.eager.Tensor(
            arr,
            core.CPUPlace(),
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True,
        )
        self.assertEqual(egr_tensor5.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor5.name)
        self.assertEqual(egr_tensor5.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor5.place.is_cpu_place())
        self.assertEqual(egr_tensor5.dtype, paddle.float32)
        self.assertEqual(egr_tensor5.stop_gradient, True)

        egr_tensor6 = core.eager.Tensor(
            arr,
            place=core.CPUPlace(),
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True,
        )
        self.assertEqual(egr_tensor6.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor6.name)
        self.assertEqual(egr_tensor6.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor6.place.is_cpu_place())
        self.assertEqual(egr_tensor6.dtype, paddle.float32)
        self.assertEqual(egr_tensor6.stop_gradient, True)

        egr_tensor7 = core.eager.Tensor(
            arr,
            place=place,
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True,
        )
        self.assertEqual(egr_tensor7.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor7.name)
        self.assertEqual(egr_tensor7.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor7.place._equals(place))
        self.assertEqual(egr_tensor7.dtype, paddle.float32)
        self.assertEqual(egr_tensor7.stop_gradient, True)

        egr_tensor8 = core.eager.Tensor(
            arr,
            place=place,
            persistable=True,
            name="new_eager_tensor",
            zero_copy=True,
            stop_gradient=False,
        )
        self.assertEqual(egr_tensor8.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor8.name)
        self.assertEqual(egr_tensor8.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor8.place._equals(place))
        self.assertEqual(egr_tensor8.dtype, paddle.float32)
        self.assertEqual(egr_tensor8.stop_gradient, False)

        egr_tensor9 = core.eager.Tensor(
            arr, place, True, True, "new_eager_tensor", stop_gradient=False
        )
        self.assertEqual(egr_tensor9.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor9.name)
        self.assertEqual(egr_tensor9.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor9.place._equals(place))
        self.assertEqual(egr_tensor9.dtype, paddle.float32)
        self.assertEqual(egr_tensor9.stop_gradient, False)

        egr_tensor10 = core.eager.Tensor(
            arr, place, True, True, name="new_eager_tensor", stop_gradient=False
        )
        self.assertEqual(egr_tensor10.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor10.name)
        self.assertEqual(egr_tensor10.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor10.place._equals(place))
        self.assertEqual(egr_tensor10.dtype, paddle.float32)
        self.assertEqual(egr_tensor10.stop_gradient, False)

        egr_tensor11 = core.eager.Tensor(
            arr,
            place,
            True,
            zero_copy=True,
            name="new_eager_tensor",
            stop_gradient=False,
        )
        self.assertEqual(egr_tensor11.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor11.name)
        self.assertEqual(egr_tensor11.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor11.place._equals(place))
        self.assertEqual(egr_tensor11.dtype, paddle.float32)
        self.assertEqual(egr_tensor11.stop_gradient, False)

        egr_tensor12 = core.eager.Tensor(
            arr,
            place,
            persistable=True,
            zero_copy=True,
            name="new_eager_tensor",
            stop_gradient=False,
        )
        self.assertEqual(egr_tensor12.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor12.name)
        self.assertEqual(egr_tensor12.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor12.place._equals(place))
        self.assertEqual(egr_tensor12.dtype, paddle.float32)
        self.assertEqual(egr_tensor12.stop_gradient, False)

        egr_tensor13 = core.eager.Tensor(
            value=arr,
            place=place,
            persistable=True,
            zero_copy=True,
            name="new_eager_tensor",
            stop_gradient=False,
        )
        self.assertEqual(egr_tensor13.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor13.name)
        self.assertEqual(egr_tensor13.shape, [4, 16, 16, 32])
        self.assertTrue(egr_tensor13.place._equals(place))
        self.assertEqual(egr_tensor13.dtype, paddle.float32)
        self.assertEqual(egr_tensor13.stop_gradient, False)

        # special case
        egr_tensor14 = core.eager.Tensor(
            dtype=core.VarDesc.VarType.FP32,
            dims=[4, 16, 16, 32],
            name="special_eager_tensor",
            type=core.VarDesc.VarType.DENSE_TENSOR,
            persistable=True,
        )
        self.assertEqual(egr_tensor14.persistable, True)
        self.assertEqual(egr_tensor14.name, "special_eager_tensor")
        self.assertEqual(egr_tensor14.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor14.dtype, paddle.float32)

        # init Tensor by Tensor
        egr_tensor15 = core.eager.Tensor(value=egr_tensor4)
        self.assertEqual(egr_tensor15.persistable, True)
        self.assertTrue("generated" in egr_tensor15.name)
        self.assertEqual(egr_tensor15.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor15.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor15.stop_gradient, True)
        self.assertTrue(
            egr_tensor15.place._equals(
                paddle.base.framework._current_expected_place()
            )
        )
        np.testing.assert_array_equal(egr_tensor15.numpy(), egr_tensor4.numpy())

        egr_tensor16 = core.eager.Tensor(
            value=egr_tensor4, name="new_eager_tensor"
        )
        self.assertEqual(egr_tensor16.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor16.name)
        self.assertEqual(egr_tensor16.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor16.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor16.stop_gradient, True)
        self.assertTrue(
            egr_tensor16.place._equals(
                paddle.base.framework._current_expected_place()
            )
        )
        np.testing.assert_array_equal(egr_tensor16.numpy(), egr_tensor4.numpy())

        egr_tensor17 = core.eager.Tensor(
            value=egr_tensor4,
            place=place,
            name="new_eager_tensor",
        )
        self.assertEqual(egr_tensor17.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor17.name)
        self.assertEqual(egr_tensor17.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor17.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor17.stop_gradient, True)
        self.assertTrue(egr_tensor17.place._equals(place))
        np.testing.assert_array_equal(egr_tensor17.numpy(), egr_tensor4.numpy())

        egr_tensor18 = core.eager.Tensor(
            egr_tensor4,
            place=place,
            name="new_eager_tensor",
        )
        self.assertEqual(egr_tensor18.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor18.name)
        self.assertEqual(egr_tensor18.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor18.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor18.stop_gradient, True)
        self.assertTrue(egr_tensor18.place._equals(place))
        np.testing.assert_array_equal(egr_tensor18.numpy(), egr_tensor4.numpy())

        egr_tensor19 = core.eager.Tensor(
            egr_tensor4,
            place,
            name="new_eager_tensor",
        )
        self.assertEqual(egr_tensor19.persistable, True)
        self.assertTrue("new_eager_tensor" in egr_tensor19.name)
        self.assertEqual(egr_tensor19.shape, egr_tensor4.shape)
        self.assertEqual(egr_tensor19.dtype, egr_tensor4.dtype)
        self.assertEqual(egr_tensor19.stop_gradient, True)
        self.assertTrue(egr_tensor19.place._equals(place))
        np.testing.assert_array_equal(egr_tensor19.numpy(), egr_tensor4.numpy())

        # init eager tensor by framework tensor
        x = np.random.rand(3, 3).astype('float32')
        t = paddle.base.Tensor()
        t.set(x, paddle.base.CPUPlace())
        egr_tensor20 = core.eager.Tensor(value=t)
        self.assertEqual(egr_tensor20.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor20.name)
        self.assertEqual(egr_tensor20.shape, [3, 3])
        self.assertEqual(egr_tensor20.dtype, paddle.float32)
        self.assertEqual(egr_tensor20.stop_gradient, True)
        self.assertTrue(
            egr_tensor20.place._equals(
                paddle.base.framework._current_expected_place()
            )
        )
        np.testing.assert_array_equal(egr_tensor20.numpy(), x)

        egr_tensor21 = core.eager.Tensor(value=t, place=place)
        self.assertEqual(egr_tensor21.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor21.name)
        self.assertEqual(egr_tensor21.shape, [3, 3])
        self.assertEqual(egr_tensor21.dtype, paddle.float32)
        self.assertEqual(egr_tensor21.stop_gradient, True)
        self.assertTrue(egr_tensor21.place._equals(place))
        np.testing.assert_array_equal(egr_tensor21.numpy(), x)

        egr_tensor22 = core.eager.Tensor(t, place=place)
        self.assertEqual(egr_tensor22.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor22.name)
        self.assertEqual(egr_tensor22.shape, [3, 3])
        self.assertEqual(egr_tensor22.dtype, paddle.float32)
        self.assertEqual(egr_tensor22.stop_gradient, True)
        self.assertTrue(egr_tensor22.place._equals(place))
        np.testing.assert_array_equal(egr_tensor22.numpy(), x)

        egr_tensor23 = core.eager.Tensor(t, place, name="from_framework_tensor")
        self.assertEqual(egr_tensor23.persistable, False)
        self.assertTrue("from_framework_tensor" in egr_tensor23.name)
        self.assertEqual(egr_tensor23.shape, [3, 3])
        self.assertEqual(egr_tensor23.dtype, paddle.float32)
        self.assertEqual(egr_tensor23.stop_gradient, True)
        self.assertTrue(egr_tensor23.place._equals(place))
        np.testing.assert_array_equal(egr_tensor23.numpy(), x)

        egr_tensor24 = core.eager.Tensor(
            value=t, place=place, name="from_framework_tensor"
        )
        self.assertEqual(egr_tensor24.persistable, False)
        self.assertTrue("from_framework_tensor" in egr_tensor24.name)
        self.assertEqual(egr_tensor24.shape, [3, 3])
        self.assertEqual(egr_tensor24.dtype, paddle.float32)
        self.assertEqual(egr_tensor24.stop_gradient, True)
        self.assertTrue(egr_tensor24.place._equals(place))
        np.testing.assert_array_equal(egr_tensor24.numpy(), x)

        # Bad usage
        # SyntaxError: positional argument follows keyword argument
        # egr_tensor25 = core.eager.Tensor(value=t, place)

    def test_constructor_with_kwargs(self):
        print("Test_constructor_with_kwargs")
        paddle.set_device("cpu")
        place_list = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            place_list.append(core.CUDAPlace(0))

        for p in place_list:
            self.constructor_with_kwargs(p)

    def test_copy_and_copy_to(self):
        print("Test_copy_and_copy_to")

        paddle.set_device("cpu")
        arr = np.ones([4, 16, 16, 32]).astype('float32')
        arr1 = np.zeros([4, 16]).astype('float32')
        arr2 = np.ones([4, 16, 16, 32]).astype('float32') + np.ones(
            [4, 16, 16, 32]
        ).astype('float32')
        tensor = paddle.to_tensor(arr, paddle.float32, core.CPUPlace())
        self.assertEqual(tensor.stop_gradient, True)
        tensor.stop_gradient = False
        print("Set persistable")
        tensor.persistable = False
        tensor1 = paddle.to_tensor(arr1, paddle.float32, core.CPUPlace())
        tensor1.persistable = True
        self.assertEqual(tensor1.stop_gradient, True)
        np.testing.assert_array_equal(tensor.numpy(), arr)
        print("Test copy_")
        tensor.copy_(tensor1, True)
        self.assertEqual(tensor.persistable, False)
        self.assertEqual(tensor.shape, [4, 16])
        self.assertEqual(tensor.dtype, paddle.float32)
        np.testing.assert_array_equal(tensor.numpy(), arr1)

        print("Test _copy_to")
        tensor2 = paddle.to_tensor(arr2, paddle.float32, core.CPUPlace())
        np.testing.assert_array_equal(tensor2.numpy(), arr2)
        self.assertTrue(tensor2.place.is_cpu_place())
        tensor2.persistable = True
        tensor2.stop_gradient = False
        if core.is_compiled_with_cuda():
            tensor3 = tensor2._copy_to(core.CUDAPlace(0), True)
            np.testing.assert_array_equal(tensor3.numpy(), arr2)
            self.assertEqual(tensor3.persistable, True)
            self.assertEqual(tensor3.stop_gradient, True)
            self.assertTrue(tensor3.place.is_gpu_place())

            tensor4 = tensor2.cuda(0, True)
            np.testing.assert_array_equal(tensor4.numpy(), arr2)
            self.assertEqual(tensor4.persistable, True)
            self.assertEqual(tensor4.stop_gradient, False)
            self.assertTrue(tensor4.place.is_gpu_place())

            tensor5 = tensor4.cpu()
            np.testing.assert_array_equal(tensor5.numpy(), arr2)
            self.assertEqual(tensor5.persistable, True)
            self.assertEqual(tensor5.stop_gradient, False)
            self.assertTrue(tensor5.place.is_cpu_place())

            tensor10 = paddle.to_tensor([1, 2, 3], place='gpu_pinned')
            tensor11 = tensor10._copy_to(core.CUDAPlace(0), True)
            np.testing.assert_array_equal(tensor10.numpy(), tensor11.numpy())
        else:
            tensor3 = tensor2._copy_to(core.CPUPlace(), True)
            np.testing.assert_array_equal(tensor3.numpy(), arr2)
            self.assertEqual(tensor3.persistable, True)
            self.assertEqual(tensor3.stop_gradient, True)
            self.assertTrue(tensor3.place.is_cpu_place())

            tensor4 = tensor2.cpu()
            np.testing.assert_array_equal(tensor4.numpy(), arr2)
            self.assertEqual(tensor4.persistable, True)
            self.assertEqual(tensor4.stop_gradient, False)
            self.assertTrue(tensor4.place.is_cpu_place())

    def test_share_buffer_to(self):
        arr = np.ones([4, 16, 16, 32]).astype('float32')
        arr1 = np.zeros([4, 16]).astype('float32')
        arr2 = np.ones([4, 16, 16, 32]).astype('float32') + np.ones(
            [4, 16, 16, 32]
        ).astype('float32')
        tensor = None
        tensor2 = None
        tensor = paddle.to_tensor(arr, paddle.float32, core.CPUPlace())
        tensor3 = core.eager.Tensor(value=tensor, place=core.CPUPlace())
        if core.is_compiled_with_cuda():
            tensor2 = paddle.to_tensor(arr2, paddle.float32, core.CUDAPlace(0))
        else:
            tensor2 = paddle.to_tensor(arr2, paddle.float32, core.CPUPlace())
        np.testing.assert_array_equal(tensor.numpy(), arr)
        np.testing.assert_array_equal(tensor2.numpy(), arr2)
        tensor2._share_buffer_to(tensor)
        np.testing.assert_array_equal(tensor.numpy(), arr2)
        np.testing.assert_array_equal(tensor2.numpy(), arr2)
        self.assertTrue(tensor._is_shared_buffer_with(tensor2))
        self.assertTrue(tensor2._is_shared_buffer_with(tensor))
        tensor._share_buffer_to(tensor3)
        np.testing.assert_array_equal(tensor3.numpy(), arr2)
        self.assertTrue(tensor3._is_shared_buffer_with(tensor))

    def test_0_size_tensor_share_buffer_to(self):
        x = paddle.rand([0, 4])
        y = paddle.rand([0, 4])
        x._share_buffer_to(y)

    def test_share_underline_tensor_to(self):
        arr = np.ones([4, 16, 16, 32]).astype('float32')
        arr1 = np.zeros([4, 16]).astype('float32')
        arr2 = np.ones([4, 16, 16, 32]).astype('float32') + np.ones(
            [4, 16, 16, 32]
        ).astype('float32')
        tensor = None
        tensor2 = None
        tensor = paddle.to_tensor(arr, paddle.float32, core.CPUPlace())
        tensor3 = core.eager.Tensor()
        if core.is_compiled_with_cuda():
            tensor2 = paddle.to_tensor(arr2, paddle.float32, core.CUDAPlace(0))
        else:
            tensor2 = paddle.to_tensor(arr2, paddle.float32, core.CPUPlace())
        np.testing.assert_array_equal(tensor.numpy(), arr)
        np.testing.assert_array_equal(tensor2.numpy(), arr2)
        tensor2._share_underline_tensor_to(tensor)
        np.testing.assert_array_equal(tensor.numpy(), arr2)
        np.testing.assert_array_equal(tensor2.numpy(), arr2)
        self.assertTrue(tensor._is_shared_underline_tensor_with(tensor2))
        self.assertTrue(tensor2._is_shared_underline_tensor_with(tensor))
        tensor._share_underline_tensor_to(tensor3)
        np.testing.assert_array_equal(tensor3.numpy(), arr2)
        self.assertTrue(tensor3._is_shared_underline_tensor_with(tensor))

    def test_properties(self):
        print("Test_properties")
        paddle.set_device("cpu")
        arr = np.ones([4, 16, 16, 32]).astype('float32')
        tensor = paddle.to_tensor(arr, paddle.float32, core.CPUPlace())
        self.assertEqual(tensor.shape, [4, 16, 16, 32])
        tensor.name = 'tensor_name_test'
        self.assertEqual(tensor.name, 'tensor_name_test')
        self.assertEqual(tensor.persistable, False)
        tensor.persistable = True
        self.assertEqual(tensor.persistable, True)
        tensor.persistable = False
        self.assertEqual(tensor.persistable, False)
        self.assertTrue(tensor.place.is_cpu_place())
        self.assertEqual(tensor._place_str, 'Place(cpu)')
        self.assertEqual(tensor.stop_gradient, True)
        tensor.stop_gradient = False
        self.assertEqual(tensor.stop_gradient, False)
        tensor.stop_gradient = True
        self.assertEqual(tensor.stop_gradient, True)
        self.assertEqual(tensor.type, core.VarDesc.VarType.DENSE_TENSOR)

    def test_global_properties(self):
        print("Test_global_properties")
        self.assertTrue(in_dygraph_mode())

    def test_place_guard(self):
        if core.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
            with paddle.base.framework._dygraph_place_guard(core.CPUPlace()):
                self.assertTrue(
                    isinstance(_current_expected_place(), type(core.CPUPlace()))
                )
        else:
            paddle.set_device("cpu")
            with paddle.base.framework._dygraph_place_guard(core.CPUPlace()):
                self.assertTrue(
                    isinstance(_current_expected_place(), type(core.CPUPlace()))
                )

    def test_value(self):
        arr = np.random.rand(4, 16, 16, 32).astype('float64')

        egr_tensor0 = core.eager.Tensor(value=arr)
        self.assertEqual(egr_tensor0.persistable, False)
        self.assertTrue("generated" in egr_tensor0.name)
        self.assertEqual(egr_tensor0.shape, [4, 16, 16, 32])
        self.assertTrue(
            egr_tensor0.place._equals(
                paddle.base.framework._current_expected_place()
            )
        )
        self.assertEqual(egr_tensor0.dtype, paddle.float64)
        self.assertEqual(egr_tensor0.stop_gradient, True)
        self.assertTrue(
            egr_tensor0.value().get_tensor()._dtype(),
            paddle.float64,
        )
        self.assertTrue(
            egr_tensor0.value().get_tensor()._place(),
            paddle.base.framework._current_expected_place(),
        )
        self.assertTrue(egr_tensor0.value().get_tensor()._is_initialized())

    def test_set_value(self):
        ori_arr = np.random.rand(4, 16, 16, 32).astype('float32')
        egr_tensor = core.eager.Tensor(value=ori_arr)
        self.assertEqual(egr_tensor.stop_gradient, True)
        self.assertEqual(egr_tensor.shape, [4, 16, 16, 32])
        np.testing.assert_array_equal(egr_tensor.numpy(), ori_arr)
        ori_place = egr_tensor.place

        new_arr = np.random.rand(4, 16, 16, 32).astype('float32')

        self.assertFalse(np.array_equal(egr_tensor.numpy(), new_arr))

        egr_tensor.set_value(new_arr)
        self.assertEqual(egr_tensor.stop_gradient, True)
        self.assertTrue(egr_tensor.place._equals(ori_place))
        self.assertEqual(egr_tensor.shape, [4, 16, 16, 32])
        np.testing.assert_array_equal(egr_tensor.numpy(), new_arr)

    def test_sharding_related_api(self):
        arr0 = np.random.rand(4, 16, 16, 32).astype('float32')
        egr_tensor1 = core.eager.Tensor(
            arr0, core.CPUPlace(), True, False, "numpy_tensor1", False
        )
        self.assertEqual(egr_tensor1._numel(), 32768)
        self.assertEqual(egr_tensor1._slice(0, 2)._numel(), 16384)

    def test_copy_gradient_from(self):
        np_x = np.random.random((2, 2))
        np_y = np.random.random((2, 2))
        x = paddle.to_tensor(np_x, dtype="float64", stop_gradient=False)
        y = paddle.to_tensor(np_y, dtype="float64")
        out = x + x
        out.backward()
        x._copy_gradient_from(y)
        np.testing.assert_array_equal(x.grad.numpy(), np_y)

    def test_clear(self):
        np_x = np.random.random((3, 8, 8))
        x = paddle.to_tensor(np_x, dtype="float64")
        self.assertTrue(x._is_initialized())
        x._clear()
        self.assertFalse(x._is_initialized())

    def test_use_gpudnn(self):
        np_x = np.random.random((3, 8, 8))

        self.assertTrue(in_dygraph_mode())
        x = paddle.to_tensor(np_x, dtype="float64")
        y = x._use_gpudnn(False)
        np.testing.assert_array_equal(x.numpy(), y.numpy())
        y = x._use_gpudnn(True)
        np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_md5sum(self):
        np_x = np.random.random((3, 8, 8))
        x = paddle.to_tensor(np_x, dtype="float64")
        y = paddle.to_tensor(np_x, dtype="float64")
        self.assertEqual(x._md5sum(), y._md5sum())
        x = paddle.to_tensor(np_x, dtype="bfloat16")
        y = paddle.to_tensor(np_x, dtype="bfloat16")
        self.assertEqual(x._md5sum(), y._md5sum())


class EagerParamBaseUsageTestCase(unittest.TestCase):
    def test_print(self):
        linear = paddle.nn.Linear(3, 3, bias_attr=False)
        print(linear.weight)

    def test_copy(self):
        linear = paddle.nn.Linear(1, 3)
        linear_copy = copy.deepcopy(linear)
        linear_copy2 = linear.weight._copy_to(core.CPUPlace(), True)
        np.testing.assert_array_equal(
            linear.weight.numpy(), linear_copy.weight.numpy()
        )
        np.testing.assert_array_equal(
            linear.weight.numpy(), linear_copy2.numpy()
        )

    def func_fp16_initilaizer(self):
        paddle.set_default_dtype("float16")
        linear1 = paddle.nn.Linear(1, 3, bias_attr=False)
        linear2 = paddle.nn.Linear(
            1,
            3,
            bias_attr=False,
            weight_attr=paddle.nn.initializer.Uniform(),
        )
        linear3 = paddle.nn.Linear(
            1,
            3,
            bias_attr=False,
            weight_attr=paddle.nn.initializer.TruncatedNormal(),
        )
        linear4 = paddle.nn.Linear(
            1,
            3,
            bias_attr=False,
            weight_attr=paddle.nn.initializer.KaimingUniform(),
        )
        res = [
            linear1.weight.numpy(),
            linear2.weight.numpy(),
            linear3.weight.numpy(),
            linear4.weight.numpy(),
        ]
        paddle.set_default_dtype("float32")
        return res

    def func_layer_helper_base(self, value):
        base = paddle.base.layer_helper_base.LayerHelperBase(
            "test_layer", "test_layer"
        )
        return paddle.to_tensor(value).numpy()

    def func_base_to_variable(self, value):
        paddle.to_tensor(value)

    def test_backward_with_single_tensor(self):
        arr4 = np.random.rand(4, 16, 16, 32).astype('float32')
        egr_tensor12 = core.eager.Tensor(arr4, core.CPUPlace())
        egr_tensor12.retain_grads()
        arr = np.ones([4, 16, 16, 32]).astype('float32')
        self.assertEqual(egr_tensor12.persistable, False)
        self.assertTrue("generated_tensor" in egr_tensor12.name)
        self.assertEqual(egr_tensor12.shape, [4, 16, 16, 32])
        self.assertEqual(egr_tensor12.dtype, paddle.float32)
        self.assertEqual(egr_tensor12.stop_gradient, True)
        self.assertTrue(egr_tensor12.place._equals(paddle.base.CPUPlace()))
        np.testing.assert_array_equal(egr_tensor12.numpy(), arr4)
        np.testing.assert_array_equal(egr_tensor12.gradient(), None)
        egr_tensor12.stop_gradient = False
        egr_tensor12.backward()
        np.testing.assert_array_equal(egr_tensor12.gradient(), arr)

    def test_set_value(self):
        linear = paddle.nn.Linear(1, 3)
        ori_place = linear.weight.place
        new_weight = np.ones([1, 3]).astype('float32')

        self.assertFalse(np.array_equal(linear.weight.numpy(), new_weight))

        linear.weight.set_value(new_weight)
        np.testing.assert_array_equal(linear.weight.numpy(), new_weight)
        self.assertTrue(linear.weight.place._equals(ori_place))


if __name__ == "__main__":
    unittest.main()
