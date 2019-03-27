# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.imperative.parallel as parallel
import paddle.fluid as fluid
import unittest


class TestImperateiveNCCLContext(unittest.TestCase):
    def test_nccl_context(self):
        strategy = parallel.ParallelStrategy()
        strategy.nranks = parallel.nranks()
        strategy.local_rank = parallel.local_rank()

        place = fluid.CUDAPlace(parallel.dev_id())

        with fluid.imperative.guard(place):
            parallel.prepare_context(strategy, place)


if __name__ == "__main__":
    unittest.main()
