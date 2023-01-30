# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import signal
<<<<<<< HEAD
import os
import sys
=======
import os, sys
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

from .manager import ElasticManager
from .manager import ElasticStatus
from .manager import ELASTIC_EXIT_CODE
from .manager import ElasticLevel
from .collective import CollectiveLauncher

from paddle.distributed.fleet.launch_utils import DistributeMode


def enable_elastic(args, distribute_mode):
<<<<<<< HEAD
    # elastic_level = os.getenv('PADDLE_ELASTIC_FAULT_TOLERANC_LEVEL')
    # if not elastic_level and (elastic_level != ElasticLevel.FAULT_TOLERANCE and
    #                          elastic_level != ElasticLevel.ELASTIC):
    #    return False

    # if distribute_mode != DistributeMode.COLLECTIVE:
=======
    #elastic_level = os.getenv('PADDLE_ELASTIC_FAULT_TOLERANC_LEVEL')
    #if not elastic_level and (elastic_level != ElasticLevel.FAULT_TOLERANCE and
    #                          elastic_level != ElasticLevel.ELASTIC):
    #    return False

    #if distribute_mode != DistributeMode.COLLECTIVE:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    #    return False

    if not args.elastic_server and not os.getenv('PADDLE_ELASTIC_SERVER'):
        return False

    if not args.job_id and not os.getenv('PADDLE_ELASTIC_JOB_ID'):
        return False

    if not args.np and not os.getenv('PADDLE_ELASTIC_NP'):
        return False

    return True


def launch_elastic(args, distribute_mode):

    server = args.elastic_server or os.getenv('PADDLE_ELASTIC_SERVER')
    srv, port = server.split(':')
    import etcd3
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    etcd_client = etcd3.client(host=srv, port=port)
    elastic = ElasticManager(args, etcd_client)

    signal.signal(signal.SIGTERM, elastic.signal_handler)
    signal.signal(signal.SIGABRT, elastic.signal_handler)
    signal.signal(signal.SIGINT, elastic.signal_handler)

    while True:

        # wait for all nodes ready to run
        elastic.wait()

        # execute pre hook action, eg: run shell
        elastic.pre_hook()

        # run self with specified launcher
        elastic.run(CollectiveLauncher)

        # keep wathing the health status of self and being notified for other's failure
        ret = elastic.watch()
        if ret == ElasticStatus.COMPLETED:
            break
        if ret == ElasticStatus.HOLD:
            continue
        if ret == ElasticStatus.EXIT:
            break
        if ret == ElasticStatus.ERROR:
            sys.exit(3)
        if ret == ElasticStatus.RESTART:
            sys.exit(ELASTIC_EXIT_CODE)

    if int(elastic.sigint) > 0:
        sys.exit(128 + int(elastic.sigint))
    else:
        sys.exit(0)
