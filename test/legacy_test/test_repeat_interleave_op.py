# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import Program, program_guard








class TestIndexSelectAPI(unittest.TestCase):
    def input_data(self):
        self.data_zero_dim_x = np.array(0.5)
        self.data_x = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )
        self.data_index = np.array([0, 1, 2, 1]).astype('int32')

    def test_repeat_interleave_api(self):
        paddle.enable_static()
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            x.desc.set_need_check_feed(False)
            index = paddle.static.data(
                name='repeats_',
                shape=[4],
                dtype='int32',
            )
            index.desc.set_need_check_feed(False)
            z = paddle.repeat_interleave(x, index, axis=1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x, 'repeats_': self.data_index},
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.repeat(self.data_x, self.data_index, axis=1)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 2:
        repeats = np.array([1, 2, 1]).astype('int32')
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype="float32")
            x.desc.set_need_check_feed(False)
            index = paddle.static.data(
                name='repeats_',
                shape=[3],
                dtype='int32',
            )
            index.desc.set_need_check_feed(False)
            z = paddle.repeat_interleave(x, index, axis=0)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={
                    'x': self.data_x,
                    'repeats_': repeats,
                },
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.repeat(self.data_x, repeats, axis=0)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        repeats = 2
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            x.desc.set_need_check_feed(False)
            z = paddle.repeat_interleave(x, repeats, axis=0)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x}, fetch_list=[z.name], return_numpy=False
            )
        expect_out = np.repeat(self.data_x, repeats, axis=0)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 3 zero_dim:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1], dtype="float32")
            x.desc.set_need_check_feed(False)
            z = paddle.repeat_interleave(x, repeats)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_zero_dim_x},
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.repeat(self.data_zero_dim_x, repeats)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 4 negative axis:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            x.desc.set_need_check_feed(False)
            index = paddle.static.data(
                name='repeats_',
                shape=[4],
                dtype='int32',
            )
            index.desc.set_need_check_feed(False)
            z = paddle.repeat_interleave(x, index, axis=-1)
            exe = base.Executor(base.CPUPlace())
            (res,) = exe.run(
                feed={'x': self.data_x, 'repeats_': self.data_index},
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.repeat(self.data_x, self.data_index, axis=-1)
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

   


if __name__ == '__main__':
    unittest.main()
