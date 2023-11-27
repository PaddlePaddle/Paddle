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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core


class TestUnzipOp(unittest.TestCase):
    def test_result(self):
        """
        For unzip op
        """
        paddle.enable_static()
        if core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            x = paddle.static.data(name='X', shape=[3, 4], dtype='float64')
            lod = paddle.static.data(name='lod', shape=[11], dtype='int64')
            output = paddle.incubate.operators.unzip(x, lod)

            input = [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
                [100.0, 200.0, 300.0, 400.0],
            ]
            lod = [0, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]

            feed = {
                'X': np.array(input).astype("float64"),
                'lod': np.array(lod).astype("int64"),
            }

            exe = base.Executor(place=place)
            exe.run(base.default_startup_program())
            res = exe.run(feed=feed, fetch_list=[output])
            out = [
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 20.0, 30.0, 40.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [100.0, 200.0, 300.0, 400.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
            out_np = np.array(out, dtype="float64")
            assert (res == out_np).all(), "output is not right"


class TestUnzipOp_Complex(unittest.TestCase):
    def test_result(self):
        """
        For unzip op
        """
        self.dtype = self.get_dtype()
        paddle.enable_static()
        prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(prog, startup_prog):
            if core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
                x = paddle.static.data(
                    name='Complex64_X', shape=[3, 4], dtype=self.dtype
                )
                lod = paddle.static.data(name='lodx', shape=[11], dtype='int64')
                output = paddle.incubate.operators.unzip(x, lod)
                input = [
                    [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j],
                    [10.0 + 10.0j, 20.0 + 20.0j, 30.0 + 30.0j, 40.0 + 40.0j],
                    [
                        100.0 + 100.0j,
                        200.0 + 200.0j,
                        300.0 + 300.0j,
                        400.0 + 400.0j,
                    ],
                ]
                lod = [0, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]

                feed = {
                    'Complex64_X': np.array(input).astype(self.dtype),
                    'lodx': np.array(lod).astype("int64"),
                }

                exe = base.Executor(place=place)
                exe.run(base.default_startup_program())
                res = exe.run(prog, feed=feed, fetch_list=[output])
                out = [
                    [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j],
                    [0.0j, 0.0j, 0.0j, 0.0j],
                    [10.0 + 10.0j, 20.0 + 20.0j, 30.0 + 30.0j, 40.0 + 40.0j],
                    [0.0j, 0.0j, 0.0j, 0.0j],
                    [0.0j, 0.0j, 0.0j, 0.0j],
                    [0.0j, 0.0j, 0.0j, 0.0j],
                    [
                        100.0 + 100.0j,
                        200.0 + 200.0j,
                        300.0 + 300.0j,
                        400.0 + 400.0j,
                    ],
                    [0.0j, 0.0j, 0.0j, 0.0j],
                    [0.0j, 0.0j, 0.0j, 0.0j],
                    [0.0j, 0.0j, 0.0j, 0.0j],
                ]
                out_np = np.array(out, dtype=self.dtype)
                assert (res == out_np).all(), "output is not right"

    def get_dtype(self):
        return np.complex64


class TestUnzipOp_Complex128(TestUnzipOp_Complex):
    def get_dtype(self):
        return np.complex128


if __name__ == '__main__':
    unittest.main()
