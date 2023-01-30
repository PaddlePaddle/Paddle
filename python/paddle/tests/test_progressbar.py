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

<<<<<<< HEAD
import random
import time
import unittest

import numpy as np
=======
import numpy as np
import unittest
import random
import time
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

from paddle.hapi.progressbar import ProgressBar


class TestProgressBar(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def prog_bar(self, num, epoch, width, verbose=1):
        for epoch in range(epoch):
            progbar = ProgressBar(num, verbose=verbose)
            values = [
                ['loss', 50.341673],
                ['acc', 0.00256],
            ]
            for step in range(1, num + 1):
                values[0][1] -= random.random() * 0.1
                values[1][1] += random.random() * 0.1
                if step % 10 == 0:
                    progbar.update(step, values)
                time.sleep(0.002)
            progbar.update(step, values)

        progbar.update(1, [['loss', int(1)]])
        progbar.update(1, [['loss', 'INF']])
        progbar.update(1, [['loss', 1e-4]])
<<<<<<< HEAD
        progbar.update(1, [['loss', np.array([1.0])]])
=======
        progbar.update(1, [['loss', np.array([1.])]])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        progbar.update(1, [['loss', np.array([1e-4])]])
        progbar.update(1, [['loss', np.array([1]).astype(np.uint16)]])
        progbar.start()

        progbar.update(0, values)
        progbar._dynamic_display = False
        progbar.update(1e4, values)

        progbar._num = None
        progbar.update(0, values)
        progbar._num = 1
        progbar.update(1 + 1e-4, values)

    def test1(self):
        self.prog_bar(50, 1, 30)

    def test2(self):
        self.prog_bar(50, 2, 30)

    def test4(self):
        self.prog_bar(50, 2, 30, verbose=2)

    def test_errors(self):
        with self.assertRaises(TypeError):
            ProgressBar(-1)


if __name__ == '__main__':
    unittest.main()
