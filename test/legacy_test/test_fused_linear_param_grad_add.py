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

import os
import re
import unittest

import numpy as np

import paddle
from paddle import _C_ops


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def promote_dtype(x):
    if x.dtype in [paddle.float16, paddle.bfloat16]:
        return x.astype(paddle.float32)
    else:
        return x


def recreate(x, multi_precision):
    if isinstance(x, (list, tuple)):
        return [recreate(item, multi_precision) for item in x]

    if x is None:
        return None

    if multi_precision:
        x = promote_dtype(x)

    return paddle.to_tensor(x.numpy())


def run_ground_truth(x, dy, dweight, dbias, multi_precision, has_bias):
    x, dy, dweight, dbias = recreate([x, dy, dweight, dbias], multi_precision)

    dweight_tmp = paddle.matmul(
        x.reshape([-1, x.shape[-1]]),
        dy.reshape([-1, dy.shape[-1]]),
        transpose_x=True,
    )
    if dweight is None:
        dweight = dweight_tmp
    else:
        assert dweight.shape == dweight_tmp.shape
        assert dweight.dtype == dweight.dtype
        dweight += dweight_tmp

    if has_bias:
        if multi_precision:
            dbias_tmp = (
                promote_dtype(dy).reshape([-1, dy.shape[-1]]).sum(axis=0)
            )
        else:
            dbias_tmp = dy.reshape([-1, dy.shape[-1]]).sum(axis=0)
        if dbias is None:
            dbias = dbias_tmp
        else:
            assert dbias.shape == dbias_tmp.shape
            assert dbias.dtype == dbias_tmp.dtype
            dbias += dbias_tmp

        return promote_dtype(dweight).numpy(), promote_dtype(dbias).numpy()
    else:
        return promote_dtype(dweight).numpy()


def run_fused_linear_param_grad_add(
    x, dy, dweight, dbias, multi_precision, has_bias
):
    dweight_new, dbias_new = _C_ops.fused_linear_param_grad_add(
        x, dy, dweight, dbias, multi_precision, has_bias
    )
    if dweight is not None:
        assert dweight_new.data_ptr() == dweight.data_ptr()
    if has_bias and dbias is not None:
        assert (
            dbias_new.data_ptr() == dbias.data_ptr()
        ), f"multi_precision={multi_precision}, has_bias={has_bias}, dbias.dtype={dbias.dtype}."
    if has_bias:
        return (
            promote_dtype(dweight_new).numpy(),
            promote_dtype(dbias_new).numpy(),
        )
    else:
        return promote_dtype(dweight_new).numpy()


class TestMainClassBase(unittest.TestCase):
    def setUp(self):
        self.shape = [3, 4, 32]
        self.output_size = 128
        self.dtype = paddle.float16

    def config(self):
        pass

    def rand(self, shape, dtype=None):
        x = np.random.randint(low=-5, high=5, size=shape)
        x = paddle.to_tensor(x)
        return x.astype(dtype or self.dtype)

    def generate_rand_inputs(
        self, has_dweight, has_dbias, multi_precision, has_bias
    ):
        x_shape = self.shape
        dy_shape = self.shape[:-1] + [self.output_size]
        dweight_shape = [self.shape[-1], self.output_size]
        dbias_shape = [self.output_size]

        x = self.rand(x_shape)
        dy = self.rand(dy_shape)
        if has_dweight:
            dweight = self.rand(dweight_shape)
            if multi_precision:
                dweight = promote_dtype(dweight)
        else:
            dweight = None

        if has_bias and has_dbias:
            dbias = self.rand(dbias_shape)
            if multi_precision:
                dbias = promote_dtype(dbias)
        else:
            dbias = None
        return x, dy, dweight, dbias

    def check_main(self, has_dweight, has_dbias, multi_precision, has_bias):
        x, dy, dweight, dbias = self.generate_rand_inputs(
            has_dweight, has_dbias, multi_precision, has_bias
        )
        res1 = run_ground_truth(
            x, dy, dweight, dbias, multi_precision, has_bias
        )
        res2 = run_fused_linear_param_grad_add(
            x, dy, dweight, dbias, multi_precision, has_bias
        )
        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            max_diff = np.max(np.abs(r1 - r2))
            self.assertLess(
                max_diff,
                1e-10,
                f"Check failed when: has_dweight={has_dweight}, has_dbias={has_dbias}, multi_precision={multi_precision}, has_bias={has_bias}",
            )

    def test_main(self):
        if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
            return

        prop = paddle.device.cuda.get_device_properties()
        cap = prop.major * 10 + prop.minor
        if self.dtype == paddle.bfloat16 and cap < 80:
            return

        if get_cuda_version() < 11060:
            return

        for has_dweight in [False, True]:
            for has_bias in [False, True]:
                for has_dbias in [False, True]:
                    for multi_precision in [False, True]:
                        self.check_main(
                            has_dweight, has_dbias, multi_precision, has_bias
                        )


class TestMainClassBF16(TestMainClassBase):
    def config(self):
        self.dtype = paddle.bfloat16


class TestMainClassFP32(TestMainClassBase):
    def config(self):
        self.dtype = paddle.float32


class TestMainClassFP64(TestMainClassBase):
    def config(self):
        self.dtype = paddle.float64


if __name__ == "__main__":
    unittest.main()
