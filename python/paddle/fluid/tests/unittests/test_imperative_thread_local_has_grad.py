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

<<<<<<< HEAD
import threading
import time
import unittest

import numpy as np

import paddle
import paddle.nn as nn


class SimpleNet(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
=======
import unittest
import paddle
import time
import paddle.nn as nn
import numpy as np
import threading
from paddle.fluid.framework import _test_eager_guard


class SimpleNet(nn.Layer):

    def __init__(self, in_dim, out_dim):
        super(SimpleNet, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class TestCases(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
    def test_main(self):
=======
    def func_main(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=self.thread_1_main))
        threads.append(threading.Thread(target=self.thread_2_main))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

<<<<<<< HEAD
=======
    def test_main(self):
        with _test_eager_guard():
            self.func_main()
        self.func_main()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == "__main__":
    unittest.main()
