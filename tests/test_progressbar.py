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
import random
import time

from progressbar import ProgressBar


class TestProgressBar(unittest.TestCase):
    def prog_bar(self, num, epoch, width, verbose=1):
        for epoch in range(epoch):
            progbar = ProgressBar(num, verbose=verbose)
            values = [
                ['loss', 50.341673],
                ['acc', 0.00256],
            ]
            for step in xrange(1, num + 1):
                values[0][1] -= random.random() * 0.1
                values[1][1] += random.random() * 0.1
                if step % 10 == 0:
                    progbar.update(step, values)
                time.sleep(0.002)
            progbar.update(step, values)

    def test1(self):
        self.prog_bar(50, 1, 30)

    def test2(self):
        self.prog_bar(50, 2, 30)

    def test4(self):
        self.prog_bar(50, 2, 30, verbose=2)


if __name__ == '__main__':
    unittest.main()
