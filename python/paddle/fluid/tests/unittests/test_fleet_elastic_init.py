#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import time
import unittest
import argparse
from warnings import catch_warnings

from paddle.distributed.fleet.elastic import enable_elastic, launch_elastic
from paddle.distributed.fleet.launch_utils import DistributeMode


class TestElasticInit(unittest.TestCase):

    def setUp(self):

        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2:4"

        self.args = Argument()

    def test_enable_elastic(self):
        result = enable_elastic(self.args, DistributeMode.COLLECTIVE)
        self.assertEqual(result, True)

    def test_launch_elastic(self):
        try:
            launch_elastic(self.args, DistributeMode.COLLECTIVE)
        except Exception as e:
            pass


if __name__ == "__main__":
    unittest.main()
