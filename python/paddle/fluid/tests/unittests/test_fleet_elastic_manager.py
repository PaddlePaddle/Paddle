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

<<<<<<< HEAD
import os
import unittest

from paddle.distributed.fleet.elastic.manager import (
    ELASTIC_AUTO_PARALLEL_EXIT_CODE,
    ElasticManager,
    LauncherInterface,
)


class MockLease:
=======
from __future__ import print_function

import os
import time
import unittest
import argparse

from paddle.distributed.fleet.elastic.manager import ElasticManager
from paddle.distributed.fleet.elastic.manager import LauncherInterface
from paddle.distributed.fleet.elastic.manager import ELASTIC_TIMEOUT
from paddle.distributed.fleet.elastic.manager import ELASTIC_AUTO_PARALLEL_EXIT_CODE


class MockLease():

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def refresh(self):
        pass


<<<<<<< HEAD
class MockKVMetadata:
    def __init__(self, key):
        self.key = key
        self.create_revision = 2
        self.mod_revision = 3
        self.version = 2
        self.lease_id = 0
        self.response_header = None


class MockEtcdClient:
=======
class MockEtcdClient:

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, lease=None):
        self._lease = lease

    def put(self, key, value, lease=None):
        pass

    def get(self, key):
<<<<<<< HEAD
        return b'0', MockKVMetadata(b"/prefix")
=======
        value = "0"
        return value, value
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def delete_prefix(self, key):
        pass

    def get_prefix(self, key_prefix):
<<<<<<< HEAD
        hosts = [
            (b"/prefix/host1", b"10.10.10.1:6001"),
            (b"/prefix/host2", b"10.10.10.2:6001"),
        ]
        return ((v, MockKVMetadata(k)) for k, v in hosts)

    def add_watch_callback(self, *args, **kwargs):
        return 0

    def add_watch_prefix_callback(self, key_prefix, callback, **kwargs):
        callback(None)
        return 0
=======
        hosts = ["10.10.10.1:6001", "10.10.10.2:6001"]
        return hosts

    def add_watch_callback(self, *args, **kwargs):
        return "host_watch"

    def add_watch_prefix_callback(self, key_prefix, callback, **kwargs):
        callback(None)
        return "host_watch"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def cancel_watch(self, watch_id):
        pass

    def delete(self, key):
<<<<<<< HEAD
        return True
=======
        pass
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def lease(self, ttl):
        if self._lease:
            return self._lease
        else:
            return MockLease()


class TestElasticManager(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.etcd_client = MockEtcdClient()

    def test_elastic_manager_init(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'

        args = Argument()

<<<<<<< HEAD
        class _MockLease:
=======
        class _MockLease():

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            def refresh(self):
                raise ValueError("valid error, this only for unittest")

        etcd_client = MockEtcdClient(lease=_MockLease())
        elastic = ElasticManager(args, etcd_client=etcd_client)

    def test_match_faulttolerance(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'

        args = Argument()
        args.ips = "10.10.10.1,10.10.10.2"
        elastic = ElasticManager(args, self.etcd_client)
        os.environ['FLAGS_START_PORT'] = "6001"

        hosts = ["10.10.10.1:6001", "10.10.10.2:6001"]
        os.environ[
<<<<<<< HEAD
            'PADDLE_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001"
=======
            'PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertEqual(elastic._match(hosts), True)

        hosts = ["10.10.10.1:6001"]
        args.ips = "10.10.10.1"
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.1:6001"
        self.assertEqual(elastic._match(hosts), False)

    def test_match_elastic(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2:4"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'

        os.environ['PADDLE_ELASTIC_TIMEOUT'] = "60"
        args = Argument()
        args.ips = "10.10.10.1,10.10.10.2,10.10.10.3,10.10.10.4"
        os.environ['FLAGS_START_PORT'] = "6001"
        os.environ[
<<<<<<< HEAD
            'DISTRIBUTED_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001,10.10.10.4:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001,10.10.10.4:6001"
=======
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001,10.10.10.4:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001,10.10.10.4:6001"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elastic = ElasticManager(args, self.etcd_client)
        hosts = ["10.10.10.1:6001", "10.10.10.2:6001"]
        self.assertEqual(elastic._match(hosts), False)

        hosts = [
<<<<<<< HEAD
            "10.10.10.1:6001",
            "10.10.10.2:6001",
            "10.10.10.3:6001",
            "10.10.10.4:6001",
=======
            "10.10.10.1:6001", "10.10.10.2:6001", "10.10.10.3:6001",
            "10.10.10.4:6001"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]
        self.assertEqual(elastic._match(hosts), True)

        hosts = ["10.10.10.1:6001", "10.10.10.2:6001", "10.10.10.3:6001"]
        self.assertEqual(elastic._match(hosts), False)

        hosts = ["10.10.10.1:6001"]
        self.assertEqual(elastic._match(hosts), False)

        args.ips = "10.10.10.1,10.10.10.2"
        os.environ[
<<<<<<< HEAD
            'DISTRIBUTED_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001"
=======
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elastic = ElasticManager(args, self.etcd_client)
        hosts = ["10.10.10.1:6001", "10.10.10.2:6001"]
        self.assertEqual(elastic._match(hosts), True)

        # TODO test timeout
<<<<<<< HEAD
        # time.sleep(60)
        # self.assertEqual(elastic._match(hosts), True)

    def test_update_hosts_for_faulttolerance(self):
=======
        #time.sleep(60)
        #self.assertEqual(elastic._match(hosts), True)

    def test_update_hosts_for_faulttolerance(self):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "0"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'

        args = Argument()
        os.environ['FLAGS_START_PORT'] = "6001"
        os.environ['PADDLE_ELASTIC_NP'] = "2"
        os.environ['PADDLE_TRAINERS'] = "10.10.10.1,10.10.10.2"
        os.environ[
<<<<<<< HEAD
            'DISTRIBUTED_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001"
=======
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elastic = ElasticManager(args, self.etcd_client)
        # add 10.10.10.3:6001
        os.environ['PADDLE_TRAINER_ID'] = "0"
        elastic.curr_host = "10.10.10.1:6001"
        elastic.hosts = ["10.10.10.1:6001", "10.10.10.2:6001"]
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.2")

        # add 10.10.10.3:6001
        elastic.curr_host = "10.10.10.3:6001"
        elastic.hosts = ["10.10.10.1:6001", "10.10.10.3:6001"]
        os.environ['PADDLE_TRAINER_ID'] = "1"
        elastic._update_hosts()
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.3")

        elastic.curr_host = "10.10.10.3:6001"
        elastic.hosts = ["10.10.10.1:6001", "10.10.10.3:6001"]
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
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'

        args = Argument()

        os.environ['FLAGS_START_PORT'] = "6001"
        os.environ['PADDLE_TRAINERS'] = "10.10.10.1,10.10.10.2"
        os.environ[
<<<<<<< HEAD
            'DISTRIBUTED_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.2:6001"
=======
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.2:6001"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elastic = ElasticManager(args, self.etcd_client)
        # add 10.10.10.3:6001
        elastic.curr_host = "10.10.10.1:6001"
        elastic.hosts = [
<<<<<<< HEAD
            "10.10.10.1:6001",
            "10.10.10.2:6001",
            "10.10.10.3:6001",
        ]
        elastic._update_hosts()
        # self.assertEqual(elastic.all_host_endpoints,
        #                 ["10.10.10.1:6001", "10.10.10.2:6001", "10.10.10.3:6001"])
        self.assertEqual(
            os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.2,10.10.10.3"
        )
=======
            "10.10.10.1:6001", "10.10.10.2:6001", "10.10.10.3:6001"
        ]
        elastic._update_hosts()
        #self.assertEqual(elastic.all_host_endpoints,
        #                 ["10.10.10.1:6001", "10.10.10.2:6001", "10.10.10.3:6001"])
        self.assertEqual(os.getenv('PADDLE_TRAINERS'),
                         "10.10.10.1,10.10.10.2,10.10.10.3")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        #######################
        # elastic, scale in #
        #######################
        os.environ[
<<<<<<< HEAD
            'PADDLE_TRAINERS'
        ] = "10.10.10.0,10.10.10.1,10.10.10.2,10.10.10.3"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'
        ] = "10.10.10.0:6000,10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'
        ] = "10.10.10.0:6000,10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001"
=======
            'PADDLE_TRAINERS'] = "10.10.10.0,10.10.10.1,10.10.10.2,10.10.10.3"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.0:6000,10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.0:6000,10.10.10.1:6001,10.10.10.2:6001,10.10.10.3:6001"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        os.environ['POD_IP'] = "10.10.10.1"
        os.environ['TRAINER_PORTS_NUM'] = "4"
        os.environ['PADDLE_TRAINER_ID'] = "1"
        os.environ['PADDLE_PORT'] = "6001"
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        # remove 10.10.10.1:6001
        elastic.curr_host = "10.10.10.1:6001"
        elastic.hosts = [
<<<<<<< HEAD
            "10.10.10.1:6001",
            "10.10.10.2:6001",
            "10.10.10.3:6001",
        ]
        elastic._update_hosts()
        # self.assertEqual(elastic.all_host_endpoints,
        #                 ["10.10.10.3:6001", "10.10.10.1:6001", "10.10.10.2:6001"])
        self.assertEqual(
            os.getenv('PADDLE_TRAINERS'), "10.10.10.3,10.10.10.1,10.10.10.2"
        )
        self.assertEqual(
            os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS'),
            "10.10.10.3:6001,10.10.10.1:6001,10.10.10.2:6001",
        )

        ############
        os.environ[
            'PADDLE_TRAINERS'
        ] = "10.10.10.1,10.10.10.1,10.10.10.1,10.10.10.1"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.1:6002,10.10.10.1:6003,10.10.10.1:6004"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'
        ] = "10.10.10.1:6001,10.10.10.1:6002,10.10.10.1:6003,10.10.10.1:6004"
=======
            "10.10.10.1:6001", "10.10.10.2:6001", "10.10.10.3:6001"
        ]
        elastic._update_hosts()
        #self.assertEqual(elastic.all_host_endpoints,
        #                 ["10.10.10.3:6001", "10.10.10.1:6001", "10.10.10.2:6001"])
        self.assertEqual(os.getenv('PADDLE_TRAINERS'),
                         "10.10.10.3,10.10.10.1,10.10.10.2")
        self.assertEqual(os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS'),
                         "10.10.10.3:6001,10.10.10.1:6001,10.10.10.2:6001")

        ############
        os.environ[
            'PADDLE_TRAINERS'] = "10.10.10.1,10.10.10.1,10.10.10.1,10.10.10.1"
        os.environ[
            'DISTRIBUTED_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.1:6002,10.10.10.1:6003,10.10.10.1:6004"
        os.environ[
            'PADDLE_TRAINER_ENDPOINTS'] = "10.10.10.1:6001,10.10.10.1:6002,10.10.10.1:6003,10.10.10.1:6004"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        os.environ['POD_IP'] = "10.10.10.1"
        os.environ['TRAINER_PORTS_NUM'] = "4"
        os.environ['PADDLE_PORT'] = "6001"
        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        # remove 10.10.10.1:6001
        elastic.curr_host = "10.10.10.1:6001"
        os.environ['PADDLE_TRAINER_ID'] = "-1"
        elastic.hosts = ["10.10.10.1:6001", "10.10.10.1:6003"]
        elastic._update_hosts()
<<<<<<< HEAD
        # self.assertEqual(elastic.all_host_endpoints,
        #                 ["10.10.10.1:6001", "10.10.10.1:6001"])
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.1")
        self.assertEqual(
            os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS'),
            "10.10.10.1:6001,10.10.10.1:6003",
        )

    def test_exit(self):
=======
        #self.assertEqual(elastic.all_host_endpoints,
        #                 ["10.10.10.1:6001", "10.10.10.1:6001"])
        self.assertEqual(os.getenv('PADDLE_TRAINERS'), "10.10.10.1,10.10.10.1")
        self.assertEqual(os.getenv('DISTRIBUTED_TRAINER_ENDPOINTS'),
                         "10.10.10.1:6001,10.10.10.1:6003")

    def test_exit(self):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'

        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.exit()

    def test_pre_hook(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
            elastic_pre_hook = None

        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.pre_hook()

        args.elastic_pre_hook = "hostname"
        elastic.pre_hook()

    def test_watch(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        class Argument:
            elastic_server = "127.0.0.1:2379"
            job_id = "test_job_id_123"
            np = "2"
            gpus = "0"
            nproc_per_node = 1
            host = None
            curr_host = None
            ips = None
            scale = None
            force = None
            backend = 'gloo'
            elastic_pre_hook = None

        class ElasticLauncher:
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            def watch(self):
                return ELASTIC_AUTO_PARALLEL_EXIT_CODE

            def stop(self):
                pass

        args = Argument()
        elastic = ElasticManager(args, self.etcd_client)
        elastic.stopped = False
        elastic.launcher = ElasticLauncher()
        elastic.watch()

    def test_launcher_interface_check_procs(self):
<<<<<<< HEAD
        class Proc:
=======

        class Proc:

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            def poll(self):
                return ELASTIC_AUTO_PARALLEL_EXIT_CODE

        class ProcList:
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            def __init__(self):
                self.proc = Proc()

        launch = LauncherInterface(None)
        launch.procs = [ProcList()]
        launch._check_procs()


if __name__ == "__main__":
    unittest.main()
