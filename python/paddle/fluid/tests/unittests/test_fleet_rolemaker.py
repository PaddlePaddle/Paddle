#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import unittest

import paddle.fluid.incubate.fleet.base.role_maker as role_maker


class TestCloudRoleMaker(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_PSERVERS"] = "127.0.0.1,127.0.0.2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"

    def test_tr_rolemaker(self):
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"

        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        ro.generate_role()

        self.assertTrue(ro.is_worker())
        self.assertFalse(ro.is_server())
        self.assertEqual(ro.worker_num(), 2)

    def test_ps_rolemaker(self):
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["POD_IP"] = "127.0.0.1"

        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        ro.generate_role()
        self.assertFalse(ro.is_worker())
        self.assertTrue(ro.is_server())
        self.assertEqual(ro.worker_num(), 2)

    def test_traing_role(self):
        os.environ["TRAINING_ROLE"] = "TEST"
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro.generate_role)


if __name__ == "__main__":
    unittest.main()
