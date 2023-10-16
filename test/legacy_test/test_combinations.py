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
from itertools import combinations, combinations_with_replacement

import numpy as np

import paddle
from paddle.base import Program

paddle.enable_static()


def convert_combinations_to_array(x, r=2, with_replacement=False):
    if r == 0:
        return np.array([]).astype(x.dtype)
    if with_replacement:
        combs = combinations_with_replacement(x, r)
    else:
        combs = combinations(x, r)
    combs = list(combs)
    res = []
    for i in range(len(combs)):
        res.append(list(combs[i]))
    return np.array(res).astype(x.dtype)


class TestCombinationsAPIBase(unittest.TestCase):
    def setUp(self):
        self.init_setting()
        self.modify_setting()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)

        self.place = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')

    def init_setting(self):
        self.dtype_np = 'float64'
        self.x_shape = [10]
        self.r = 5
        self.with_replacement = False

    def modify_setting(self):
        pass

    def test_static_graph(self):
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(Program()):
                x = paddle.static.data(
                    name="x", shape=self.x_shape, dtype=self.dtype_np
                )
                out = paddle.combinations(x, self.r, self.with_replacement)
                exe = paddle.static.Executor(place=place)
                feed_list = {"x": self.x_np}
                pd_res = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_list,
                    fetch_list=[out],
                )[0]
                ref_res = convert_combinations_to_array(
                    self.x_np, self.r, self.with_replacement
                )
                np.testing.assert_allclose(ref_res, pd_res, atol=1e-5)

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_pd = paddle.to_tensor(self.x_np)
            pd_res = paddle.combinations(x_pd, self.r, self.with_replacement)
            ref_res = convert_combinations_to_array(
                self.x_np, self.r, self.with_replacement
            )
            np.testing.assert_allclose(ref_res, pd_res, atol=1e-5)

    def test_errors(self):
        def test_input_not_1D():
            data_np = np.random.random((10, 10)).astype(np.float32)
            res = paddle.combinations(data_np, self.r, self.with_replacement)

        self.assertRaises(TypeError, test_input_not_1D)

        def test_r_range():
            res = paddle.combinations(self.x_np, -1, self.with_replacement)

        self.assertRaises(ValueError, test_r_range)


class TestIndexFillAPI1(TestCombinationsAPIBase):
    def modify_setting(self):
        self.dtype_np = 'int32'
        self.x_shape = [10]
        self.r = 1
        self.with_replacement = True


class TestIndexFillAPI2(TestCombinationsAPIBase):
    def modify_setting(self):
        self.dtype_np = 'int64'
        self.x_shape = [10]
        self.r = 0
        self.with_replacement = True


class TestIndexFillAPI3(TestCombinationsAPIBase):
    def modify_setting(self):
        self.dtype_np = 'float32'
        self.x_shape = [0]
        self.r = 10
        self.with_replacement = False
