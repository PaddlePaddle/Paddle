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

import os
import unittest
import numpy as np

from paddle.text.datasets import *


class TestConll05st(unittest.TestCase):
    def test_main(self):
        conll05st = Conll05st()
        self.assertTrue(len(conll05st) == 5267)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 5267)
        sample = conll05st[idx]
        self.assertTrue(len(sample) == 9)
        for s in sample:
            self.assertTrue(len(s.shape) == 1)

        assert os.path.exists(conll05st.get_embedding())


if __name__ == '__main__':
    unittest.main()
