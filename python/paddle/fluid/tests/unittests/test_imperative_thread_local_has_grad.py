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

import threading
import time
import unittest

import numpy as np

import paddle
from paddle import nn


class SimpleNet(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class TestCases(unittest.TestCase):
    @paddle.no_grad()
    def thread_1_main(self):
        time.sleep(8)

    def thread_2_main(self):
        in_dim = 10
        out_dim = 3
        net = SimpleNet(in_dim, out_dim)
        for _ in range(1000):
            x = paddle.to_tensor(np.random.rand(32, in_dim).astype('float32'))
            self.assertTrue(x.stop_gradient)
            x = net(x)
            self.assertFalse(x.stop_gradient)

    def test_main(self):
        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=self.thread_1_main))
        threads.append(threading.Thread(target=self.thread_2_main))
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    unittest.main()
