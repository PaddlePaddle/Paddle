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
from paddle.base import core
from paddle.static import Program, program_guard

DYNAMIC = 1
STATIC = 2


def _run_masked_fill(mode, x, mask, value, device='cpu'):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
        # Set device
        paddle.set_device(device)
        x_ = paddle.to_tensor(x)
        mask_ = paddle.to_tensor(mask)
        # value is scaler
        if isinstance(value, (float, int)):
            value_ = value
        # value is tensor
        else:
            value_ = paddle.to_tensor(value)
        res = paddle.masked_fill(x_, mask_, value_)
        return res.numpy()
    # static graph mode
    elif mode == STATIC:
        paddle.enable_static()
        # value is scalar
        if isinstance(value, (float, int)):
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                mask_ = paddle.static.data(
                    name="mask", shape=mask.shape, dtype=mask.dtype
                )
                value_ = value
                res = paddle.masked_fill(x_, mask_, value_)
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
                exe = paddle.static.Executor(place)
                outs = exe.run(
                    feed={'x': x, 'mask': mask, 'value': value},
                    fetch_list=[res],
                )
                return outs[0]
        # y is tensor
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                mask_ = paddle.static.data(
                    name="mask", shape=mask.shape, dtype=mask.dtype
                )
                value_ = paddle.static.data(
                    name="value", shape=value.shape, dtype=value.dtype
                )
                res = paddle.masked_fill(x_, mask_, value_)
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
                exe = paddle.static.Executor(place)
                outs = exe.run(
                    feed={'x': x, 'mask': mask, 'value': value},
                    fetch_list=[res],
                )
                return outs[0]


def check_dtype(input, desired_dtype):
    if input.dtype != desired_dtype:
        raise ValueError(
            "The expected data type to be obtained is {}, but got {}".format(
                desired_dtype, input.dtype
            )
        )


def _np_masked_fill(x, mask, value):
    y = np.full_like(x, value)
    return np.where(mask, y, x)


class TestMaskedFillAPI(unittest.TestCase):
    def setUp(self):
        self.places = ['cpu']
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_masked_fill(self):
        np.random.seed(7)
        for place in self.places:
            shape = (100, 100)
            for dt in (np.float64, np.float32, np.int64, np.int32):
                x = np.random.uniform((-5), 5, shape).astype(dt)
                mask = np.random.randint(2, size=shape).astype('bool')
                value = np.random.uniform((-5), 5)
                res = _run_masked_fill(DYNAMIC, x, mask, value, place)
                check_dtype(res, dt)
                np.testing.assert_allclose(res, _np_masked_fill(x, mask, value))
                res = _run_masked_fill(STATIC, x, mask, value, place)
                check_dtype(res, dt)
                np.testing.assert_allclose(res, _np_masked_fill(x, mask, value))
                # broadcast
                x = np.random.uniform((-5), 5, shape).astype(dt)
                mask = np.random.randint(2, size=shape[1:]).astype('bool')
                value = np.random.uniform((-5), 5)
                res = _run_masked_fill(DYNAMIC, x, mask, value, place)
                check_dtype(res, dt)
                np.testing.assert_allclose(res, _np_masked_fill(x, mask, value))
                res = _run_masked_fill(STATIC, x, mask, value, place)
                check_dtype(res, dt)
                np.testing.assert_allclose(res, _np_masked_fill(x, mask, value))


if __name__ == '__main__':
    unittest.main()
