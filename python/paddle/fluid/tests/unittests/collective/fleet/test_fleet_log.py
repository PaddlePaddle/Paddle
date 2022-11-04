# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.log_util import logger
import logging
import unittest


class TestFleetLog(unittest.TestCase):

    def setUp(self):
        fleet.init(log_level="DEBUG")

    def test_log_level(self):

        # check correctly initialized
        assert fleet.get_log_level_code() == logging._nameToLevel["DEBUG"]
        assert logger.getEffectiveLevel() == logging._nameToLevel["DEBUG"]

        # test set name
        fleet.set_log_level("WARNING")
        debug1 = fleet.get_log_level_code()
        debug2 = logging._nameToLevel["WARNING"]
        assert debug1 == debug2

        # test set int
        fleet.set_log_level(debug2)

        # check the logger is changed
        assert logger.getEffectiveLevel() == logging._nameToLevel["WARNING"]
