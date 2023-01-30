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
import unittest
=======
from __future__ import print_function

import os
import unittest
import paddle.fluid as fluid
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

from test_parallel_dygraph_dataparallel import TestMultipleGpus


class TestParallelMarginSoftmaxWithCrossEntropy(TestMultipleGpus):
<<<<<<< HEAD
    def test_parallel_margin_cross_entropy(self):
        self.run_mnist_2gpu('parallel_margin_cross_entropy.py')


if __name__ == "__main__":
=======

    def test_parallel_margin_cross_entropy(self):
        self.run_mnist_2gpu('parallel_margin_cross_entropy.py')
        self.run_mnist_2gpu('parallel_margin_cross_entropy.py',
                            eager_mode=False)


if __name__ == "__main__":
    os.environ["FLAGS_enable_eager_mode"] = "1"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    unittest.main()
