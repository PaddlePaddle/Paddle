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

from __future__ import print_function

import unittest
import time
import paddle.fluid as fluid

from paddle.distributed.utils import find_free_ports, watch_local_trainers, get_cluster, start_local_trainers


def get_cluster_from_args(selected_gpus):
    cluster_node_ips = '127.0.0.1'
    node_ip = '127.0.0.1'

    node_ips = [x.strip() for x in cluster_node_ips.split(',')]

    node_ips.index(node_ip)

    free_ports = None

    free_ports = find_free_ports(len(selected_gpus))
    if free_ports is not None:
        free_ports = list(free_ports)

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus)


def get_gpus(selected_gpus):
    selected_gpus = [x.strip() for x in selected_gpus.split(',')]
    return selected_gpus


class TestMultipleGpus(unittest.TestCase):
    def run_mnist_2gpu(self, target_file_name):
        if not fluid.core.is_compiled_with_cuda(
        ) or fluid.core.get_cuda_device_count() == 0:
            return

        selected_gpus = get_gpus('0,1')
        cluster = None
        pod = None

        cluster, pod = get_cluster_from_args(selected_gpus)

        procs = start_local_trainers(
            cluster,
            pod,
            training_script=target_file_name,
            training_script_args=[])

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_nranks())

            if not alive:
                print("Local procs complete, POD info:{}".format(pod))
                break
            time.sleep(3)

    def test_multiple_gpus_dynamic(self):
        self.run_mnist_2gpu('parallel_dygraph_gradient_check.py')


if __name__ == "__main__":
    unittest.main()
