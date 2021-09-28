# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.device.cuda.graphs import CUDAGraph
import unittest
import numpy as np


class TestCUDAGraph(unittest.TestCase):
    def random_tensor(self, shape):
        return paddle.to_tensor(
            np.random.randint(
                low=0, high=10, size=shape).astype("float32"))

    def test_cuda_graph(self):
        if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
            return

        shape = [2, 3]
        x = self.random_tensor(shape)
        z = self.random_tensor(shape)

        g = CUDAGraph()
        g.capture_begin()
        y = x + 10
        z.add_(x)
        g.capture_end()

        for _ in range(10):
            z_np_init = z.numpy()
            x_new = self.random_tensor(shape)
            x.copy_(x_new, False)
            g.replay()
            x_np = x_new.numpy()
            y_np = y.numpy()
            z_np = z.numpy()
            self.assertTrue((y_np - x_np == 10).all())
            self.assertTrue((z_np - z_np_init == x_np).all())

        g.reset()


if __name__ == "__main__":
    unittest.main()
