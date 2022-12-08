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

import paddle
import paddle.fluid as fluid


class TestOpNameConflict(unittest.TestCase):
    def test_conflict(self):
        paddle.enable_static()
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                x = fluid.data(name="x", shape=[1], dtype='float32')
                y = fluid.data(name="y", shape=[1], dtype='float32')

                m = paddle.log2(x, name="log2")
                n = paddle.log2(y, name="log2")

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                m_v, n_v = exe.run(
                    feed={
                        "x": np.ones((1), "float32") * 1,
                        "y": np.ones((1), "float32") * 2,
                    },
                    fetch_list=[m, n],
                )

                self.assertEqual(m_v[0], 0.0)
                self.assertEqual(n_v[0], 1.0)


if __name__ == '__main__':
    unittest.main()
