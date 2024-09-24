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

import unittest

import numpy as np
from decorator_helper import prog_scope

import paddle
from paddle import base
from paddle.framework import in_pir_mode


class TestMathOpPatches(unittest.TestCase):
    @classmethod
    def setUp(self):
        np.random.seed(1024)
        paddle.enable_static()

    @prog_scope()
    def test_equal_and_cond(self):
        a = paddle.static.data(name="a", shape=[-1, 1], dtype='float32')
        b = paddle.static.data(name="b", shape=[-1, 1], dtype='float32')
        if not in_pir_mode():
            a.desc.set_need_check_feed(False)
            b.desc.set_need_check_feed(False)
        one = paddle.ones(shape=[1], dtype='int32')
        zero = paddle.zeros(shape=[1], dtype='int32')
        cond = one == zero
        c = paddle.static.nn.cond(cond, lambda: a + b, lambda: a - b)

        place = base.CPUPlace()
        exe = base.Executor(place)
        a_np = np.array([3, 4, 10, 14, 9, 18]).astype('float32')
        b_np = np.array([3, 4, 11, 15, 8, 18]).astype('float32')

        (c_np,) = exe.run(
            paddle.static.default_main_program(),
            feed={"a": a_np, "b": b_np},
            fetch_list=[c],
        )

        np.testing.assert_array_equal(c_np, a_np - b_np)


if __name__ == '__main__':
    unittest.main()
