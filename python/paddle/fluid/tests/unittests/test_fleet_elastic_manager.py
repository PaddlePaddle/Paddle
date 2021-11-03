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
from warnings import catch_warnings

from paddle.distributed.fleet.elastic.manager import ElasticManager
from paddle.distributed.fleet.elastic.manager import ELASTIC_TIMEOUT


class TestElasticManager(unittest.TestCase):
    def setUp(self):
        class MockEtcdClient:
            def put(self, key, value):
                pass

            def get(self, key):
                value = "0"
                return value, value

            def delete_prefix(self, key):
                pass

            def get_prefix(self, key_prefix):
                hosts = ["10.10.10.1", "10.10.10.2"]
                return hosts

            def add_watch_callback(self, *args, **kwargs):
                return "host_watch"

            def cancel_watch(self, watch_id):
                pass

            def delete(self, key):
                pass

        self.etcd_client = MockEtcdClient()

    def test_match_faulttolerance(self):
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2"
            host = None
            scale = None
            force = None

        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        hosts = ["10.10.10.1", "10.10.10.2"]
        self.assertEqual(elastic._match(hosts), True)

    def test_match_elastic(self):
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2:4"
            host = None
            scale = None
            force = None

        os.environ['PADDLE_ELASTIC_TIMEOUT'] = "60"
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        hosts = ["10.10.10.1", "10.10.10.2"]
        self.assertEqual(elastic._match(hosts), False)

        hosts = ["10.10.10.1", "10.10.10.2", "10.10.10.3"]
        self.assertEqual(elastic._match(hosts), False)

        hosts = ["10.10.10.1"]
        self.assertEqual(elastic._match(hosts), False)

        # TODO test timeout
        #time.sleep(60)
        #self.assertEqual(elastic._match(hosts), True)

    def test_update_hosts_for_faulttolerance(self):
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2"
            host = None
            scale = None
            force = None

        args = Argument()
        os.environ['PADDLE_TRAINERS'] = "10.10.10.1,10.10.10.2"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:8001,10.10.10.2:8001"
        elastic = ElasticManager(args, self.etcd_client)
        # add 10.10.10.3
        os.environ['PADDLE_TRAINER_ID'] = "0"
        elastic.host = "10.10.10.1"
        elastic.hosts = ["10.10.10.1", "10.10.10.2"]
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.2")

        # add 10.10.10.3
        elastic.host = "10.10.10.3"
        elastic.hosts = ["10.10.10.1", "10.10.10.3"]
        os.environ['PADDLE_TRAINER_ID'] = "1"
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.3")

        elastic.host = "10.10.10.3"
        elastic.hosts = ["10.10.10.1", "10.10.10.3"]
        os.environ['PADDLE_TRAINER_ID'] = "-1"
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.3")

    def test_update_hosts_for_elastic(self):
        #######################
        #  elastic, scale up  #
        #######################
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2:4"
            host = None
            scale = None
            force = None

        args = Argument()

        os.environ['PADDLE_TRAINERS'] = "10.10.10.1,10.10.10.2"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:8001,10.10.10.2:8001"
        elastic = ElasticManager(args, self.etcd_client)
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
        elastic = ElasticManager(args, self.etcd_client)
        # remove 10.10.10.1
        elastic.host = "10.10.10.1"
        elastic.hosts = ["10.10.10.1", "10.10.10.2", "10.10.10.3"]
        elastic._update_hosts()
        self.assertEqual(elastic.lastest_trainers,
                         "10.10.10.3,10.10.10.1,10.10.10.2")
        self.assertEqual(
            os.getenv('PADDLE_TRAINERS'), "10.10.10.3,10.10.10.1,10.10.10.2")

        ############
        os.environ['PADDLE_TRAINERS'] = "10.10.10.1,10.10.10.1"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:8001,10.10.10.1:8002,10.10.10.1:8003,10.10.10.1:8004"
        elastic = ElasticManager(args, self.etcd_client)
        # remove 10.10.10.1
        elastic.host = "10.10.10.1"
        os.environ['PADDLE_TRAINER_ID'] = "-1"
        elastic.hosts = ["10.10.10.1", "10.10.10.1"]
        elastic._update_hosts()
        self.assertEqual(elastic.lastest_trainers, "10.10.10.1,10.10.10.1")
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.1")


if __name__ == "__main__":
    unittest.main()
