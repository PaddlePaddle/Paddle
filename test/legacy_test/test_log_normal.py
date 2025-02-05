#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

np.random.seed(10)
paddle.seed(10)


def log_normal_mean(mean, std):
    return np.exp(mean + np.power(std, 2) / 2.0)


def log_normal_var(mean, std):
    var = np.power(std, 2)
    return (np.exp(var) - 1.0) * np.exp(2 * mean + var)


class TestLogNormalAPI(unittest.TestCase):
    def setUp(self):
        self.mean = 0.0
        self.std = 0.5
        self.shape = None
        self.mean_duplicate = None
        self.std_duplicate = None
        self.duplicates = 1000
        self.set_attrs()
        self.dtype = self.get_dtype()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def set_attrs(self):
        self.shape = [self.duplicates]

    def get_shape(self):
        if isinstance(self.mean, np.ndarray):
            shape = self.mean_duplicate.shape
        elif isinstance(self.std, np.ndarray):
            shape = self.std_duplicate.shape
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
        shape = self.get_shape()
        main_program = paddle.static.Program()
        if isinstance(self.mean, np.ndarray) and isinstance(
            self.std, np.ndarray
        ):
            with paddle.static.program_guard(main_program):
                mean = paddle.static.data(
                    'Mean', self.mean_duplicate.shape, self.mean_duplicate.dtype
                )
                std = paddle.static.data(
                    'Std', self.std_duplicate.shape, self.std_duplicate.dtype
                )
                out = paddle.log_normal(mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                ret = exe.run(
                    feed={
                        'Mean': self.mean_duplicate,
                        'Std': self.std_duplicate.reshape(shape),
                    },
                    fetch_list=[out],
                )
                return ret[0]
        elif isinstance(self.mean, np.ndarray):
            with paddle.static.program_guard(main_program):
                mean = paddle.static.data(
                    'Mean', self.mean_duplicate.shape, self.mean_duplicate.dtype
                )
                out = paddle.log_normal(mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                ret = exe.run(
                    feed={'Mean': self.mean_duplicate}, fetch_list=[out]
                )
            return ret[0]
        elif isinstance(self.std, np.ndarray):
            with paddle.static.program_guard(main_program):
                std = paddle.static.data(
                    'Std', self.std_duplicate.shape, self.std_duplicate.dtype
                )
                out = paddle.log_normal(self.mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                ret = exe.run(
                    feed={'Std': self.std_duplicate}, fetch_list=[out]
                )
            return ret[0]
        else:
            with paddle.static.program_guard(main_program):
                out = paddle.log_normal(self.mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                ret = exe.run(fetch_list=[out])
            return ret[0]

    def dygraph_api(self):
        paddle.disable_static(self.place)
        shape = self.get_shape()

        mean = (
            paddle.to_tensor(self.mean_duplicate)
            if isinstance(self.mean, np.ndarray)
            else self.mean
        )
        std = (
            paddle.to_tensor(self.std_duplicate)
            if isinstance(self.std, np.ndarray)
            else self.std
        )
        out = paddle.log_normal(mean, std, self.shape)
        ret = out.numpy()
        paddle.enable_static()
        return ret

    def test_api(self):
        paddle.enable_static()
        ret_static = self.static_api()
        ret_dygraph = self.dygraph_api()
        for ret in [ret_static, ret_dygraph]:
            shape_ref = self.get_shape()
            self.assertEqual(shape_ref, list(ret.shape))

            mean = np.mean(ret, axis=0, keepdims=True)
            var = np.var(ret, axis=0, keepdims=True)
            mean_ref = log_normal_mean(self.mean, self.std)
            var_ref = log_normal_var(self.mean, self.std)
            np.testing.assert_allclose(mean_ref, mean, rtol=0.2, atol=0.2)
            np.testing.assert_allclose(var_ref, var, rtol=0.2, atol=0.2)


class TestLogNormalAPI_mean_is_tensor(TestLogNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(-0.5, -0.1, [1, 2]).astype('float64')
        self.mean_duplicate = np.broadcast_to(self.mean, [self.duplicates, 2])
        self.std = 0.5


class TestLogNormalAPI_std_is_tensor(TestLogNormalAPI):
    def set_attrs(self):
        self.std = np.random.uniform(0.1, 0.5, [1, 2]).astype('float64')
        self.std_duplicate = np.broadcast_to(self.std, [self.duplicates, 2])


class TestLogNormalAPI_mean_std_are_tensor(TestLogNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(0.1, 0.5, [1, 2]).astype('float64')
        self.mean_duplicate = np.broadcast_to(self.mean, [self.duplicates, 2])
        self.std = np.random.uniform(0.1, 0.5, [1, 2]).astype('float64')
        self.std_duplicate = np.broadcast_to(self.std, [self.duplicates, 2])


class TestLogNormalAlias(unittest.TestCase):

    def test_alias(self):
        paddle.disable_static()
        shape = [1, 2, 3]
        out1 = paddle.log_normal(shape=shape)
        out2 = paddle.tensor.log_normal(shape=shape)
        out3 = paddle.tensor.random.log_normal(shape=shape)
        paddle.enable_static()


class TestLogNormalErrors(unittest.TestCase):

    def test_errors(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            mean = [1, 2, 3]
            self.assertRaises(TypeError, paddle.log_normal, mean)

            std = [1, 2, 3]
            self.assertRaises(TypeError, paddle.log_normal, std=std)

            mean = paddle.static.data('Mean', [100], 'int32')
            self.assertRaises(TypeError, paddle.log_normal, mean)

            std = paddle.static.data('Std', [100], 'int32')
            self.assertRaises(TypeError, paddle.log_normal, mean=1.0, std=std)

            self.assertRaises(TypeError, paddle.log_normal, shape=1)

            self.assertRaises(TypeError, paddle.log_normal, shape=[1.0])

            shape = paddle.static.data('Shape', [100], 'float32')
            self.assertRaises(TypeError, paddle.log_normal, shape=shape)


if __name__ == "__main__":
    unittest.main()
