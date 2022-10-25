# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

import unittest
import numpy as np


class TestDLPack(unittest.TestCase):
    def test_paddle_dlpack(self):
        a = np.random.randint(low=0, high=15, size=[10, 10])
        b = paddle.to_tensor(a)
        dlpack = paddle.utils.dlpack.to_dlpack(b)
        c = paddle.utils.dlpack.from_dlpack(dlpack)
        np.testing.assert_allclose(a, c.numpy(), rtol=1e-05, atol=1e-06)

    def test_dlpack_deletion(self):
        # See Paddle issue 47171
        for i in range(100):
            a = paddle.rand(shape=[1024 * 128, 1024], dtype="float32")
            dlpack = paddle.utils.dlpack.to_dlpack(a)
            c = paddle.utils.dlpack.from_dlpack(dlpack)


if __name__ == "__main__":
    unittest.main()
