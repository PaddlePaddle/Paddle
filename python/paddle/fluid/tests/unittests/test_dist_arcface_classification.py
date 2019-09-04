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
from test_dist_classification_base import TestDistClassificationBase


class TestDistArcfaceClassification(TestDistClassificationBase):
    def test_training(self):
        if fluid.core.is_compiled_with_cuda():
            self.compare_parall_to_local(
                'dist_arcface_classification.py', delta=1e-5)


class TestDistArcfaceClassificationParam(TestDistClassificationBase):
    def append_common_cmd(self):
        return '--arcface_margin 0.5 --arcface_scale 64'

    def test_training(self):
        if fluid.core.is_compiled_with_cuda():
            self.compare_parall_to_local(
                "dist_arcface_classification.py", delta=1e-5)


if __name__ == "__main__":
    unittest.main()
