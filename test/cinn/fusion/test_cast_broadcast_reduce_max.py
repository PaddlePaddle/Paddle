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
            'eager_in_tmp_8': self.random([32, 1, 1, 128], "float32")
        }

    def build_program(self, builder, target):
        eager_in_tmp_8 = builder.create_input(
            self.nptype2cinntype(self.feed_data['eager_in_tmp_8'].dtype),
            self.feed_data['eager_in_tmp_8'].shape,
            "eager_in_tmp_8",
        )

        var_15 = builder.cast(eager_in_tmp_8, dtype="float16")
        # cast should not fused into reduce when the output need fetch
        var_73 = builder.broadcast_to(
            var_15, broadcast_axes=[0, 1, 2, 3], out_shape=[32, 12, 128, 128]
        )
        var_55 = builder.cast(var_73, dtype="float32")
        var_76 = builder.reduce_max(var_55, dim=[3], keep_dim=False)

        return [eager_in_tmp_8], [var_15, var_76]

    def test_check_results(self):
        self.check_fusion_outputs(group_size=2)


class TestGroup2(FusionTest):
    def init_input_data(self):
        self.feed_data = {
            'eager_in_tmp_8': self.random([32, 1, 1, 128], "float32")
        }

    def build_program(self, builder, target):
        eager_in_tmp_8 = builder.create_input(
            self.nptype2cinntype(self.feed_data['eager_in_tmp_8'].dtype),
            self.feed_data['eager_in_tmp_8'].shape,
            "eager_in_tmp_8",
        )

        var_15 = builder.cast(eager_in_tmp_8, dtype="float16")
        # cast should  fused into reduce when the output not fetched
        var_73 = builder.broadcast_to(
            var_15, broadcast_axes=[0, 1, 2, 3], out_shape=[32, 12, 128, 128]
        )
        var_55 = builder.cast(var_73, dtype="float32")
        var_76 = builder.reduce_max(var_55, dim=[3], keep_dim=False)

        return [eager_in_tmp_8], [var_76]

    def test_check_results(self):
        self.check_fusion_outputs(group_size=1)


if __name__ == "__main__":
    unittest.main()
