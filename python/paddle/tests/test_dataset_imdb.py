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
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import numpy as np

from paddle.text.datasets import Imdb


class TestImdbTrain(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_main(self):
        imdb = Imdb(mode='train')
        self.assertTrue(len(imdb) == 25000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 25000)
        data, label = imdb[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(int(label) in [0, 1])


class TestImdbTest(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_main(self):
        imdb = Imdb(mode='test')
        self.assertTrue(len(imdb) == 25000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 25000)
        data, label = imdb[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(int(label) in [0, 1])


if __name__ == '__main__':
    unittest.main()
