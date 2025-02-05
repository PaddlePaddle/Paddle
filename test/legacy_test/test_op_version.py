# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import base, utils


class OpLastCheckpointCheckerTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.checker = utils.OpLastCheckpointChecker()
        self.fake_op = 'for_pybind_test__'

    def test_op_attr_info(self):
        update_type = base.core.OpUpdateType.kNewAttr
        info_list = self.checker.filter_updates(
            self.fake_op, update_type, 'STRINGS'
        )
        self.assertTrue(info_list)
        self.assertEqual(info_list[0].name(), 'STRINGS')
        self.assertEqual(info_list[0].default_value(), ['str1', 'str2'])
        self.assertEqual(info_list[0].remark(), 'std::vector<std::string>')

    def test_op_input_output_info(self):
        update_type = base.core.OpUpdateType.kNewInput
        info_list = self.checker.filter_updates(
            self.fake_op, update_type, 'NewInput'
        )
        self.assertTrue(info_list)
        self.assertEqual(info_list[0].name(), 'NewInput')
        self.assertEqual(info_list[0].remark(), 'NewInput_')

    def test_op_bug_fix_info(self):
        update_type = base.core.OpUpdateType.kBugfixWithBehaviorChanged
        info_list = self.checker.filter_updates(self.fake_op, update_type)
        self.assertTrue(info_list)
        self.assertEqual(info_list[0].remark(), 'BugfixWithBehaviorChanged_')


class OpVersionTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.vmap = base.core.get_op_version_map()
        self.fake_op = 'for_pybind_test__'

    def test_checkpoints(self):
        version_id = self.vmap[self.fake_op].version_id()
        checkpoints = self.vmap[self.fake_op].checkpoints()
        self.assertEqual(version_id, 4)
        self.assertEqual(len(checkpoints), 4)
        self.assertEqual(checkpoints[2].note(), 'Note 2')
        desc_1 = checkpoints[1].version_desc().infos()
        self.assertEqual(desc_1[0].info().default_value(), True)
        self.assertAlmostEqual(desc_1[1].info().default_value(), 1.23, 2)
        self.assertEqual(desc_1[2].info().default_value(), -1)
        self.assertEqual(desc_1[3].info().default_value(), 'hello')
        desc_2 = checkpoints[2].version_desc().infos()
        self.assertEqual(desc_2[0].info().default_value(), [True, False])
        true_l = [2.56, 1.28]
        self.assertEqual(len(true_l), len(desc_2[1].info().default_value()))
        for i in range(len(true_l)):
            self.assertAlmostEqual(
                desc_2[1].info().default_value()[i], true_l[i], 2
            )
        self.assertEqual(desc_2[2].info().default_value(), [10, 100])
        self.assertEqual(
            desc_2[3].info().default_value(), [10000001, -10000001]
        )


if __name__ == '__main__':
    unittest.main()
