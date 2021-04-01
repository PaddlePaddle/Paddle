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

from __future__ import print_function

import unittest
import time
import paddle.fluid as fluid

from paddle.distributed.utils import find_free_ports, watch_local_trainers, get_cluster, get_gpus, start_local_trainers
from test_parallel_dygraph_model_parallel import get_cluster_from_args, TestMultipleGpus


class TestMultipleGpus(TestMultipleGpus):
    def test_multiple_gpus_dynamic(self):
        self.run_mnist_2gpu('parallel_dygraph_parallel_random.py')


if __name__ == "__main__":
    unittest.main()
