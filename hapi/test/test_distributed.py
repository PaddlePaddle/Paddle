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
import os
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle
import paddle.fluid as fluid

from paddle.distributed.utils import *
import paddle.distributed.cloud_utils as cloud_utils


def get_cluster_from_args(selected_gpus):
    cluster_node_ips = '127.0.0.1'
    node_ip = '127.0.0.1'
    use_paddlecloud = False
    started_port = None
    node_ips = [x.strip() for x in cluster_node_ips.split(',')]

    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args:node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    free_ports = None
    if not use_paddlecloud and len(node_ips) <= 1 and started_port is None:
        free_ports = find_free_ports(len(selected_gpus))
        if free_ports is not None:
            free_ports = list(free_ports)
    else:
        started_port = 6070

        free_ports = [
            x for x in range(started_port, started_port + len(selected_gpus))
        ]
    return get_cluster(node_ips, node_ip, free_ports, selected_gpus)


def get_gpus(selected_gpus):
    if selected_gpus is None:
        gpus_num = fluid.core.get_cuda_device_count()
        selected_gpus = [str(x) for x in range(0, gpus_num)]
    else:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or cuda_visible_devices == "":
            selected_gpus = [x.strip() for x in selected_gpus.split(',')]
        else:
            cuda_visible_devices_list = cuda_visible_devices.split(',')
            for x in selected_gpus.split(','):
                assert x in cuda_visible_devices_list, "Can't find "\
                "your selected_gpus %s in CUDA_VISIBLE_DEVICES[%s]."\
                % (x, cuda_visible_devices)
            selected_gpus = [
                cuda_visible_devices_list.index(x.strip())
                for x in selected_gpus.split(',')
            ]
    return selected_gpus


class TestMultipleGpus(unittest.TestCase):
    def test_mnist_2gpu(self):
        if fluid.core.get_cuda_device_count() == 0:
            return

        selected_gpus = get_gpus('0,1')
        cluster = None
        pod = None

        cluster, pod = get_cluster_from_args(selected_gpus)

        procs = start_local_trainers(
            cluster,
            pod,
            training_script='dist_mnist.py',
            training_script_args=[])

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_nranks())

            if not alive:
                logger.info("Local procs complete, POD info:{}".format(pod))
                break
            time.sleep(3)


if __name__ == "__main__":
    unittest.main()
