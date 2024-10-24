#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from itertools import combinations

import numpy as np

import paddle
from paddle.base import Program

paddle.enable_static()


def compute_index_fill_ref(x, axis, index, value):
    perm = list(range(len(x.shape)))
    perm[0] = axis
    perm[axis] = 0

    out = np.transpose(x, perm)
    out[index] = value
    out = np.transpose(out, perm)
    return out


class TestIndexFillAPIBase(unittest.TestCase):
    def setUp(self):
        self.init_setting()
        self.modify_setting()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.index_np = np.array(self.combs[np.random.randint(0, 252)]).astype(
            self.index_type
        )

        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.place.append('cpu')
        if self.dtype_np == 'float16':
            self.place = []
        if paddle.is_compiled_with_cuda():
            self.place.append('gpu')

    def init_setting(self):
        self.dtype_np = 'float64'
        self.index_type = 'int64'
        self.x_shape = (20, 40)
        self.index_size = (5,)
        self.axis = 0
        self.value = -1
        self.combs = list(combinations(list(range(10)), self.index_size[0]))

    def modify_setting(self):
        pass

    def test_static_graph(self):
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(Program()):
                x = paddle.static.data(
                    name="x", shape=self.x_shape, dtype=self.dtype_np
                )
                index = paddle.static.data(
                    name="index", shape=self.index_size, dtype=self.index_type
                )
                out = paddle.index_fill(x, index, self.axis, self.value)
                exe = paddle.static.Executor(place=place)
                feed_list = {"x": self.x_np, "index": self.index_np}
                pd_res = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_list,
                    fetch_list=[out],
                )[0]
                ref_res = compute_index_fill_ref(
                    self.x_np, self.axis, self.index_np, self.value
                )
                np.testing.assert_allclose(ref_res, pd_res)

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_pd = paddle.to_tensor(self.x_np)
            index_pd = paddle.to_tensor(self.index_np)
            pd_res = paddle.index_fill(x_pd, index_pd, self.axis, self.value)
            ref_res = compute_index_fill_ref(
                self.x_np, self.axis, self.index_np, self.value
            )
            np.testing.assert_allclose(ref_res, pd_res)

    def test_errors(self):
        data_np = np.random.random((10, 10)).astype(np.float32)
        index = paddle.to_tensor([0, 2])

        def test_index_not_tensor():
            res = paddle.index_fill(data_np, [0, 2], axis=-1, value=-1)

        self.assertRaises(ValueError, test_index_not_tensor)

        def test_value_shape():
            res = paddle.index_fill(
                data_np, index, axis=-1, value=paddle.to_tensor([-1, -4])
            )

        self.assertRaises(ValueError, test_value_shape)

        def test_axis_range():
            res = paddle.index_fill(data_np, index, axis=4, value=-1)

        self.assertRaises(ValueError, test_axis_range)


class TestIndexFillAPI1(TestIndexFillAPIBase):
    def modify_setting(self):
        self.dtype_np = 'int64'
        self.index_type = 'int32'
        self.x_shape = (10, 15, 10)
        self.axis = 1


class TestIndexFillAPI2(TestIndexFillAPIBase):
    def modify_setting(self):
        self.dtype_np = 'bool'
        self.index_type = 'int32'
        self.x_shape = (10, 15, 10)
        self.axis = 1
        self.value = True


class TestIndexFillAPI3(TestIndexFillAPIBase):
    def modify_setting(self):
        self.dtype_np = 'float16'
        self.x_shape = (10, 15, 10)
        self.axis = 1
        self.value = 0.5


if __name__ == "__main__":
    unittest.main()
