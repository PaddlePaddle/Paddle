# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from test_dist_collective_base import TestDistCollectiveBase


class TestDistSoftmaxClassification(TestDistCollectiveBase):
    def test_training(self):
        if fluid.core.is_compiled_with_cuda():
            self.compare_parall_to_local(
                "dist_softmax_classification.py", delta=1e-5)


if __name__ == "__main__":
    unittest.main()
