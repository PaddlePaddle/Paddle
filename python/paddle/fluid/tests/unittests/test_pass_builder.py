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

import unittest

import paddle.fluid as fluid


class TestPassBuilder(unittest.TestCase):
    def test_parallel_testing_with_new_strategy(self):
        build_strategy = fluid.BuildStrategy()
        self.assertFalse(build_strategy.fuse_elewise_add_act_ops)
        build_strategy.fuse_elewise_add_act_ops = True
        # FIXME: currently fuse_elewise_add_act_ops not compatible with below options
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False
        pass_builder = build_strategy._finalize_strategy_and_create_passes()
        self.assertTrue(
            "fuse_elewise_add_act_pass"
            in [p.type() for p in pass_builder.all_passes()]
        )

        origin_len = len(pass_builder.all_passes())

        viz_pass = pass_builder.append_pass("graph_viz_pass")
        self.assertEqual(origin_len + 1, len(pass_builder.all_passes()))

        pass_builder.insert_pass(
            len(pass_builder.all_passes()), "graph_viz_pass"
        )
        self.assertEqual(origin_len + 2, len(pass_builder.all_passes()))

        pass_builder.remove_pass(len(pass_builder.all_passes()) - 1)
        self.assertEqual(origin_len + 1, len(pass_builder.all_passes()))


if __name__ == '__main__':
    unittest.main()
