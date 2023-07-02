# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

from fusion_test import FusionTest


class TestGroup1(FusionTest):
    def init_input_data(self):
        self.feed_data = {}

    def build_program(self, builder, target):
        x = builder.fill_constant(
            dtype="float32", shape=[4, 5, 20, 20], value=1.00000000
        )
        y = builder.cast(
            builder.reduce_sum(x, dim=[2], keep_dim=False), "float16"
        )

        feed_list = []
        fetch_list = [y]

        return feed_list, fetch_list

    def test_check_results(self):
        self.check_fusion_outputs(group_size=1)


if __name__ == "__main__":
    unittest.main()
