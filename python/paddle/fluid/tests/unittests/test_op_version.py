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

from __future__ import print_function

import unittest

import paddle.utils as utils
import paddle.fluid as fluid


class OpLastCheckpointCheckerTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(OpLastCheckpointCheckerTest, self).__init__(methodName)
        self.checker = utils.OpLastCheckpointChecker()

    def test_op_attr_info(self):
        update_type = fluid.core.OpUpdateType.kNewAttr
        info_list = self.checker.filter_updates('arg_max', update_type,
                                                'flatten')
        self.assertTrue(info_list)
        self.assertTrue(info_list[0].name())
        self.assertTrue(info_list[0].default_value() == False)
        self.assertTrue(info_list[0].remark())

    def test_op_input_output_info(self):
        update_type = fluid.core.OpUpdateType.kNewInput
        info_list = self.checker.filter_updates('roi_align', update_type,
                                                'RoisNum')
        self.assertTrue(info_list)
        self.assertTrue(info_list[0].name())
        self.assertTrue(info_list[0].remark())

    def test_op_bug_fix_info(self):
        update_type = fluid.core.OpUpdateType.kBugfixWithBehaviorChanged
        info_list = self.checker.filter_updates('leaky_relu', update_type)
        self.assertTrue(info_list)
        self.assertTrue(info_list[0].remark())


class OpVersionTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(OpVersionTest, self).__init__(methodName)
        self.vmap = fluid.core.get_op_version_map()

    def test_checkpoints(self):
        version_id = self.vmap['arg_max'].version_id()
        checkpoints = self.vmap['arg_max'].checkpoints()
        self.assertTrue(version_id)
        self.assertTrue(checkpoints)
        self.assertTrue(checkpoints[0].note())
        self.assertTrue(checkpoints[0].version_desc().infos())


if __name__ == '__main__':
    unittest.main()
