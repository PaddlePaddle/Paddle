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
        self.feed_data = {
            'cond': self.random([1, 1, 100, 100], "bool"),
            'true_value': self.random([1, 1, 100, 100], "float64"),
            'false_value': self.random([1, 1, 100, 100], "float64"),
        }

    def build_program(self, builder, target):
        cond = builder.create_input(
            self.nptype2cinntype(self.feed_data['cond'].dtype),
            self.feed_data['cond'].shape,
            "cond",
        )
        true_value = builder.create_input(
            self.nptype2cinntype(self.feed_data['true_value'].dtype),
            self.feed_data['true_value'].shape,
            "true_value",
        )
        false_value = builder.create_input(
            self.nptype2cinntype(self.feed_data['false_value'].dtype),
            self.feed_data['false_value'].shape,
            "false_value",
        )

        var_1 = builder.select(cond, true_value, false_value)
        var_2 = builder.reduce_sum(var_1, dim=[2], keep_dim=False)

        feed_list = [cond, true_value, false_value]
        fetch_list = [var_2]

        return feed_list, fetch_list

    def test_check_results(self):
        self.check_fusion_outputs(group_size=1)


if __name__ == "__main__":
    unittest.main()
