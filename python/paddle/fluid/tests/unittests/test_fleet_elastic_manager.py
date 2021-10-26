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

from __future__ import print_function

import os
import time
import unittest
import argparse

from paddle.distributed.fleet.elastic.manager import ElasticManager
from paddle.distributed.fleet.elastic.manager import ELASTIC_TIMEOUT


class TestElasticManager(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser(description='Elastic Command')
        parser.add_argument(
            "--elastic_server",
            type=str,
            default="127.0.0.1:2379",
            help="etcd server host:port")
        parser.add_argument(
            "--job_id",
            type=str,
            default="test_job_id_123",
            help="job unique id")
        parser.add_argument(
            "--np",
            type=str,
            default="2:4",
            help="job pod/node number, need to be 'MIN' or 'MIN:MAX' format")
        parser.add_argument("--host", type=str, default=None, help="host")
        parser.add_argument("--scale", type=int, default=None, help="scale")
        parser.add_argument("--force", type=str, default=None, help="force")
        self.args = parser.parse_args()

    def test_match(self):
        elastic = ElasticManager(self.args)
        hosts = ["10.10.10.1", "10.10.10.2"]
        self.assertEqual(elastic._match(hosts), False)

        hosts = ["10.10.10.1", "10.10.10.2", "10.10.10.3"]
        self.assertEqual(elastic._match(hosts), False)

        # TODO test timeout
        #time.sleep(60)
        #self.assertEqual(elastic._match(hosts), True)

    def test_update_hosts(self):
        #######################
        #  elastic, scale up  #
        #######################
        os.environ['PADDLE_TRAINERS'] = "10.10.10.1,10.10.10.2"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:8001,10.10.10.2:8001"
        elastic = ElasticManager(self.args)
        # add 10.10.10.3
        elastic.host = "10.10.10.1"
        elastic.hosts = ["10.10.10.1", "10.10.10.2", "10.10.10.3"]
        elastic._update_hosts()
        self.assertEqual(elastic.lastest_trainers,
                         "10.10.10.1,10.10.10.2,10.10.10.3")
        self.assertEqual(
            os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.2,10.10.10.3")

        #######################
        # elastic, scale down #
        #######################
        os.environ[
            'PADDLE_TRAINERS'] = "10.10.10.0,10.10.10.1,10.10.10.2,10.10.10.3"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.0:8001,10.10.10.1:8001,10.10.10.2:8001,10.10.10.3:8001"
        elastic = ElasticManager(self.args)
        # remove 10.10.10.1
        elastic.host = "10.10.10.1"
        elastic.hosts = ["10.10.10.1", "10.10.10.2", "10.10.10.3"]
        elastic._update_hosts()
        self.assertEqual(elastic.lastest_trainers,
                         "10.10.10.3,10.10.10.1,10.10.10.2")
        self.assertEqual(
            os.getenv('PADDLE_TRAINERS'), "10.10.10.3,10.10.10.1,10.10.10.2")


if __name__ == "__main__":
    unittest.main()
