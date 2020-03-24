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

import paddle.fluid as fluid
import numpy as np
import unittest


class TestOpNameConflict(unittest.TestCase):
    def test_conflict(self):
        x = fluid.data(name="x", shape=[1], dtype='float32')
        y = fluid.data(name="y", shape=[1], dtype='float32')
        z = fluid.data(name="z", shape=[1], dtype='float32')

        m = fluid.layers.elementwise_add(x, y, name="add")
        n = fluid.layers.elementwise_add(y, z, name="add")
        p = m + n

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        m_v, n_v, p_v = exe.run(feed={
            "x": np.ones((1), "float32") * 2,
            "y": np.ones((1), "float32") * 3,
            "z": np.ones((1), "float32") * 5
        },
                                fetch_list=[m, n, p])

        self.assertEqual(m_v[0], 5.0)
        self.assertEqual(n_v[0], 8.0)
        self.assertEqual(p_v[0], 13.0)


if __name__ == '__main__':
    unittest.main()
