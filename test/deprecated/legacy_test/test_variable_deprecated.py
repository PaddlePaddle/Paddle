#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import (
    default_main_program,
)

paddle.enable_static()


class TestVariable(unittest.TestCase):
    def setUp(self):
        np.random.seed(2022)

    def _test_slice(self, place):
        b = default_main_program().current_block()
        w = b.create_var(dtype="float64", shape=[784, 100, 100])

        for i in range(3):
            nw = w[i]
            self.assertEqual((100, 100), nw.shape)

        nw = w[:]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[:, :]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[:, :, -1]
        self.assertEqual((784, 100), nw.shape)

        nw = w[1, 1, 1]

        self.assertEqual(len(nw.shape), 0)

        nw = w[:, :, :-1]
        self.assertEqual((784, 100, 99), nw.shape)

        self.assertEqual(0, nw.lod_level)

        main = base.Program()
        with base.program_guard(main):
            exe = base.Executor(place)
            tensor_array = np.array(
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                    [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
                ]
            ).astype('float32')
            var = paddle.assign(tensor_array)
            var1 = var[0, 1, 1]
            var2 = var[1:]
            var3 = var[0:1]
            var4 = var[::-1]
            var5 = var[1, 1:, 1:]
            var_reshape = paddle.reshape(var, [3, -1, 3])
            var6 = var_reshape[:, :, -1]
            var7 = var[:, :, :-1]
            var8 = var[:1, :1, :1]
            var9 = var[:-1, :-1, :-1]
            var10 = var[::-1, :1, :-1]
            var11 = var[:-1, ::-1, -1:]
            var12 = var[1:2, 2:, ::-1]
            var13 = var[2:10, 2:, -2:-1]
            var14 = var[1:-1, 0:2, ::-1]
            var15 = var[::-1, ::-1, ::-1]

            x = paddle.static.data(name='x', shape=[-1, 13], dtype='float32')
            y = paddle.static.nn.fc(x, size=1, activation=None)
            y_1 = y[:, 0]
            feeder = base.DataFeeder(place=place, feed_list=[x])
            data = []
            data.append(np.random.randint(10, size=[13]).astype('float32'))
            exe.run(base.default_startup_program())

            local_out = exe.run(
                main,
                feed=feeder.feed([data]),
                fetch_list=[
                    var,
                    var1,
                    var2,
                    var3,
                    var4,
                    var5,
                    var6,
                    var7,
                    var8,
                    var9,
                    var10,
                    var11,
                    var12,
                    var13,
                    var14,
                    var15,
                ],
            )

            np.testing.assert_array_equal(local_out[1], tensor_array[0, 1, 1:2])
            np.testing.assert_array_equal(local_out[2], tensor_array[1:])
            np.testing.assert_array_equal(local_out[3], tensor_array[0:1])
            np.testing.assert_array_equal(local_out[4], tensor_array[::-1])
            np.testing.assert_array_equal(local_out[5], tensor_array[1, 1:, 1:])
            np.testing.assert_array_equal(
                local_out[6], tensor_array.reshape((3, -1, 3))[:, :, -1]
            )
            np.testing.assert_array_equal(local_out[7], tensor_array[:, :, :-1])
            np.testing.assert_array_equal(
                local_out[8], tensor_array[:1, :1, :1]
            )
            np.testing.assert_array_equal(
                local_out[9], tensor_array[:-1, :-1, :-1]
            )
            np.testing.assert_array_equal(
                local_out[10], tensor_array[::-1, :1, :-1]
            )
            np.testing.assert_array_equal(
                local_out[11], tensor_array[:-1, ::-1, -1:]
            )
            np.testing.assert_array_equal(
                local_out[12], tensor_array[1:2, 2:, ::-1]
            )
            np.testing.assert_array_equal(
                local_out[13], tensor_array[2:10, 2:, -2:-1]
            )
            np.testing.assert_array_equal(
                local_out[14], tensor_array[1:-1, 0:2, ::-1]
            )
            np.testing.assert_array_equal(
                local_out[15], tensor_array[::-1, ::-1, ::-1]
            )

    def test_slice(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self._test_slice(place)


if __name__ == '__main__':
    unittest.main()
