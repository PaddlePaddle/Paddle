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

from __future__ import division

import sys
import unittest
import numpy as np

import paddle
from ..unittests.test_multiprocess_dataloader_static import TestStaticDataLoader

paddle.enable_static()


class TestStaticDataLoader(TestStaticDataLoader):
    def test_main(self):
        results = []
        places = [paddle.NPUPlace(0)]

        for num_workers in [0, 2]:
            print(self.__class__.__name__, places, num_workers)
            sys.stdout.flush()
            ret = self._run_main(
                num_workers=num_workers, places=places, use_pe=False)
            results.append(ret)

        diff = np.max(
            np.abs(results[0]['loss'] - results[1]['loss']) /
            np.abs(results[0]['loss']))
        self.assertLess(diff, 1e-2)


if __name__ == '__main__':
    unittest.main()
