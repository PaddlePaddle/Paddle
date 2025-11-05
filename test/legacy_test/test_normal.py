#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place, is_custom_device

import paddle

paddle.enable_static()

np.random.seed(10)
paddle.seed(10)


class TestNormalAPI(unittest.TestCase):
    def setUp(self):
        self.mean = 1.0
        self.std = 0.0
        self.shape = None
        self.repeat_num = 2000
        self.set_attrs()
        self.dtype = self.get_dtype()
        self.place = (
            get_device_place()
            if (
                (paddle.base.core.is_compiled_with_cuda() or is_custom_device())
                or is_custom_device()
            )
            else paddle.CPUPlace()
        )

    def set_attrs(self):
        self.shape = [8, 12]

    def get_shape(self):
        if isinstance(self.mean, np.ndarray):
            shape = self.mean.shape
        elif isinstance(self.std, np.ndarray):
            shape = self.std.shape
        else:
            shape = self.shape
        return list(shape)

    def get_dtype(self):
        if isinstance(self.mean, np.ndarray):
            return self.mean.dtype
        elif isinstance(self.std, np.ndarray):
            return self.std.dtype
        else:
            return 'float32'

    def static_api(self):
        paddle.enable_static()
        shape = self.get_shape()
        ret_all_shape = copy.deepcopy(shape)
        ret_all_shape.insert(0, self.repeat_num)
        ret_all = np.zeros(ret_all_shape, self.dtype)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            if isinstance(self.mean, np.ndarray) and isinstance(
                self.std, np.ndarray
            ):
                mean = paddle.static.data(
                    'Mean', self.mean.shape, self.mean.dtype
                )
                std = paddle.static.data('Std', self.std.shape, self.std.dtype)
                out = paddle.normal(mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(
                        feed={
                            'Mean': self.mean,
                            'Std': self.std.reshape(shape),
                        },
                        fetch_list=[out],
                    )
                    ret_all[i] = ret[0]
            elif isinstance(self.mean, np.ndarray):
                mean = paddle.static.data(
                    'Mean', self.mean.shape, self.mean.dtype
                )
                out = paddle.normal(mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(feed={'Mean': self.mean}, fetch_list=[out])
                    ret_all[i] = ret[0]
            elif isinstance(self.std, np.ndarray):
                std = paddle.static.data('Std', self.std.shape, self.std.dtype)
                out = paddle.normal(self.mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(feed={'Std': self.std}, fetch_list=[out])
                    ret_all[i] = ret[0]
            else:
                out = paddle.normal(self.mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(fetch_list=[out])
                    ret_all[i] = ret[0]
        paddle.disable_static()
        return ret_all

    def dygraph_api(self):
        paddle.disable_static(self.place)
        shape = self.get_shape()
        ret_all_shape = copy.deepcopy(shape)
        ret_all_shape.insert(0, self.repeat_num)
        ret_all = np.zeros(ret_all_shape, self.dtype)

        mean = (
            paddle.to_tensor(self.mean)
            if isinstance(self.mean, np.ndarray)
            else self.mean
        )
        std = (
            paddle.to_tensor(self.std)
            if isinstance(self.std, np.ndarray)
            else self.std
        )
        for i in range(self.repeat_num):
            out = paddle.normal(mean, std, self.shape)
            ret_all[i] = out.numpy()
        paddle.enable_static()
        return ret_all

    def test_api(self):
        ret_static = self.static_api()
        ret_dygraph = self.dygraph_api()
        for ret in [ret_static, ret_dygraph]:
            shape_ref = self.get_shape()
            self.assertEqual(shape_ref, list(ret[0].shape))

            ret = ret.flatten().reshape([self.repeat_num, -1])
            mean = np.mean(ret, axis=0)
            std = np.std(ret, axis=0)
            mean_ref = (
                self.mean.flatten()
                if isinstance(self.mean, np.ndarray)
                else self.mean
            )
            std_ref = (
                self.std.flatten()
                if isinstance(self.std, np.ndarray)
                else self.std
            )
            np.testing.assert_allclose(mean_ref, mean, rtol=0.2, atol=0.2)
            np.testing.assert_allclose(std_ref, std, rtol=0.2, atol=0.2)


class TestNormalAPI_mean_is_tensor(TestNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(-2, -1, [2, 3, 4, 5]).astype('float64')


class TestNormalAPI_std_is_tensor(TestNormalAPI):
    def set_attrs(self):
        self.std = np.random.uniform(0.7, 1, [2, 3, 17]).astype('float64')


class TestNormalAPI_mean_std_are_tensor(TestNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(1, 2, [1, 100]).astype('float64')
        self.std = np.random.uniform(0.5, 1, [1, 100]).astype('float64')


class TestNormalAPI_mean_std_are_tensor_with_different_dtype(TestNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(1, 2, [100]).astype('float64')
        self.std = np.random.uniform(1, 2, [100]).astype('float32')


class TestNormalAlias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        shape = [1, 2, 3]
        out1 = paddle.normal(shape=shape)
        out2 = paddle.tensor.normal(shape=shape)
        out3 = paddle.tensor.random.normal(shape=shape)
        paddle.enable_static()


class TestNormalErrors(unittest.TestCase):
    def test_errors(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            mean = [1, 2, 3]
            self.assertRaises(TypeError, paddle.normal, mean)

            std = [1, 2, 3]
            self.assertRaises(TypeError, paddle.normal, std=std)

            mean = paddle.static.data('Mean', [100], 'int32')
            self.assertRaises(TypeError, paddle.normal, mean)

            std = paddle.static.data('Std', [100], 'int32')
            self.assertRaises(TypeError, paddle.normal, mean=1.0, std=std)

            self.assertRaises(TypeError, paddle.normal, shape=1)
            self.assertRaises(TypeError, paddle.normal, shape=[1.0])

            shape = paddle.static.data('Shape', [100], 'float32')
            self.assertRaises(TypeError, paddle.normal, shape=shape)


class TestNormalAPIComplex(unittest.TestCase):
    def setUp(self):
        self.mean = 1.0 + 1.0j
        self.std = 1.0
        self.shape = None
        self.repeat_num = 2000
        self.set_attrs()
        self.dtype = self.get_dtype()
        self.place = (
            get_device_place()
            if (
                (paddle.base.core.is_compiled_with_cuda() or is_custom_device())
                or is_custom_device()
            )
            else paddle.CPUPlace()
        )

    def set_attrs(self):
        self.shape = [8, 12]

    def get_shape(self):
        if isinstance(self.mean, np.ndarray):
            shape = self.mean.shape
        elif isinstance(self.std, np.ndarray):
            shape = self.std.shape
        else:
            shape = self.shape
        return list(shape)

    def get_dtype(self):
        if isinstance(self.mean, np.ndarray):
            return self.mean.dtype
        else:
            return 'complex64'

    def static_api(self):
        paddle.enable_static()
        shape = self.get_shape()
        ret_all_shape = copy.deepcopy(shape)
        ret_all_shape.insert(0, self.repeat_num)
        ret_all = np.zeros(ret_all_shape, self.dtype)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            if isinstance(self.mean, np.ndarray) and isinstance(
                self.std, np.ndarray
            ):
                mean = paddle.static.data(
                    'Mean', self.mean.shape, self.mean.dtype
                )
                std = paddle.static.data('Std', self.std.shape, self.std.dtype)
                out = paddle.normal(mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(
                        feed={
                            'Mean': self.mean,
                            'Std': self.std.reshape(shape),
                        },
                        fetch_list=[out],
                    )
                    ret_all[i] = ret[0]
            elif isinstance(self.mean, np.ndarray):
                mean = paddle.static.data(
                    'Mean', self.mean.shape, self.mean.dtype
                )
                out = paddle.normal(mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(feed={'Mean': self.mean}, fetch_list=[out])
                    ret_all[i] = ret[0]
            elif isinstance(self.std, np.ndarray):
                mean = paddle.static.data('Mean', self.std.shape, 'complex128')
                std = paddle.static.data('Std', self.std.shape, self.std.dtype)
                out = paddle.normal(mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(
                        feed={
                            'Std': self.std,
                            'Mean': np.broadcast_to(
                                np.array(self.mean), self.std.shape
                            ),
                        },
                        fetch_list=[out],
                    )
                    ret_all[i] = ret[0]
            else:
                mean = paddle.static.data('Mean', (), 'complex128')
                out = paddle.normal(mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(
                        feed={'Mean': np.array(self.mean)}, fetch_list=[out]
                    )
                    ret_all[i] = ret[0]
        paddle.disable_static()
        return ret_all

    def dygraph_api(self):
        paddle.disable_static(self.place)
        shape = self.get_shape()
        ret_all_shape = copy.deepcopy(shape)
        ret_all_shape.insert(0, self.repeat_num)
        ret_all = np.zeros(ret_all_shape, self.dtype)

        mean = (
            paddle.to_tensor(self.mean)
            if isinstance(self.mean, np.ndarray)
            else self.mean
        )
        std = (
            paddle.to_tensor(self.std)
            if isinstance(self.std, np.ndarray)
            else self.std
        )
        for i in range(self.repeat_num):
            out = paddle.normal(mean, std, self.shape)
            ret_all[i] = out.numpy()
        paddle.enable_static()
        return ret_all

    def test_api(self):
        ret_static = self.static_api()
        ret_dygraph = self.dygraph_api()
        for ret in [ret_static, ret_dygraph]:
            shape_ref = self.get_shape()
            self.assertEqual(shape_ref, list(ret[0].shape))

            mean = np.mean(ret, axis=0)
            var = np.var(ret, axis=0)
            var_real = np.var(ret.real, axis=0)
            var_imag = np.var(ret.imag, axis=0)
            mean_ref = self.mean
            var_ref = np.power(self.std, 2)
            np.testing.assert_allclose(mean_ref, mean, rtol=0.2, atol=0.2)
            np.testing.assert_allclose(var_ref, var, rtol=0.2, atol=0.2)
            np.testing.assert_allclose(
                var_ref / 2.0, var_real, rtol=0.2, atol=0.2
            )
            np.testing.assert_allclose(
                var_ref / 2.0, var_imag, rtol=0.2, atol=0.2
            )


class TestNormalAPIComplex_mean_is_tensor(TestNormalAPIComplex):
    def set_attrs(self):
        self.mean = np.vectorize(complex)(
            np.random.uniform(-2, -1, [2, 3]), np.random.uniform(-2, -1, [2, 3])
        )


class TestNormalAPIComplex_std_is_tensor(TestNormalAPIComplex):
    def set_attrs(self):
        self.std = np.random.uniform(0.7, 1, [2, 5]).astype('float64')


class TestNormalAPIComplex_mean_std_are_tensor(TestNormalAPIComplex):
    def set_attrs(self):
        self.mean = np.vectorize(complex)(
            np.random.uniform(1, 2, [1, 100]), np.random.uniform(1, 2, [1, 100])
        )
        self.std = np.random.uniform(0.5, 1, [1, 100]).astype('float64')


class TestNormalAPIComplex_mean_std_are_tensor_with_different_dtype(
    TestNormalAPIComplex
):
    def set_attrs(self):
        self.mean = np.vectorize(complex)(
            np.random.uniform(1, 2, [100]), np.random.uniform(1, 2, [100])
        ).astype('complex64')
        self.std = np.random.uniform(1, 2, [100]).astype('float32')


class TestNormalComplexErrors(unittest.TestCase):
    def test_errors(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            mean = 1 + 1j
            self.assertRaises(TypeError, paddle.normal, mean, dtype='float32')

            mean = 2 + 0.5j
            self.assertRaises(ValueError, paddle.normal, mean)


class TestNormalAPI_out_parameter(unittest.TestCase):
    def test_out_with_shape(self):
        paddle.disable_static()
        shape = [2, 3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=0.0, std=1.0, shape=shape, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_mean_tensor(self):
        paddle.disable_static()

        mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=mean_tensor, std=1.0, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_std_tensor(self):
        paddle.disable_static()

        std_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=0.0, std=std_tensor, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_mean_std_tensors(self):
        paddle.disable_static()

        mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        std_tensor = paddle.to_tensor([0.5, 1.0, 1.5])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=mean_tensor, std=std_tensor, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_complex_mean(self):
        paddle.disable_static()

        shape = [2, 3]

        out_tensor = paddle.empty(shape, dtype='complex64')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(
            mean=1.0 + 1.0j, std=1.0, shape=shape, out=out_tensor
        )

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)
        self.assertEqual(result.dtype, paddle.complex64)

        paddle.enable_static()

    def test_out_with_complex_mean_tensor(self):
        paddle.disable_static()

        mean_tensor = paddle.to_tensor([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='complex64')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=mean_tensor, std=1.0, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)
        self.assertEqual(result.dtype, paddle.complex64)

        paddle.enable_static()


class TestNormalAPI_size_alias(unittest.TestCase):
    def test_size_alias_basic(self):
        paddle.disable_static()
        shape = [2, 3]
        out = paddle.normal(mean=0.0, std=1.0, size=shape)
        self.assertEqual(list(out.shape), shape)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
