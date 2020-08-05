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
import paddle
import os


class TestFleetUtil(unittest.TestCase):
    def test_util_base(self):
        import paddle.fleet as fleet
        util = fleet.UtilBase()
        strategy = fleet.DistributedStrategy()
        util._set_strategy(strategy)
        role_maker = None  # should be fleet.PaddleCloudRoleMaker()
        util._set_role_maker(role_maker)

    def test_util_factory(self):
        import paddle.fleet as fleet
        factory = fleet.base.util_factory.UtilFactory()
        strategy = fleet.DistributedStrategy()
        role_maker = None  # should be fleet.PaddleCloudRoleMaker()
        optimize_ops = []
        params_grads = []
        context = {}
        context["role_maker"] = role_maker
        context["valid_strategy"] = strategy
        util = factory._create_util(context)
        self.assertEqual(util.role_maker, None)

    def test_get_util(self):
        import paddle.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        default_util = fleet.util
        self.assertEqual(default_util, None)

    def test_set_user_defined_util(self):
        import paddle.fleet as fleet

        class UserDefinedUtil(fleet.UtilBase):
            def __init__(self):
                super(UserDefinedUtil, self).__init__()

            def get_user_id(self):
                return 10

        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        my_util = UserDefinedUtil()
        fleet.util = my_util
        user_id = fleet.util.get_user_id()
        self.assertEqual(user_id, 10)


if __name__ == "__main__":
    unittest.main()
