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

import copy
import os
import subprocess
import time
import unittest

import paddle
import paddle.fluid as fluid
from paddle.distributed.utils.launch_utils import (
    TrainerProc,
    find_free_ports,
    get_cluster,
    watch_local_trainers,
)


def get_cluster_from_args(selected_xpus):
    cluster_node_ips = '127.0.0.1'
    node_ip = '127.0.0.1'

    node_ips = [x.strip() for x in cluster_node_ips.split(',')]

    node_ips.index(node_ip)

    free_ports = None

    free_ports = find_free_ports(len(selected_xpus))
    if free_ports is not None:
        free_ports = list(free_ports)

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_xpus)


def get_xpus(selected_xpus):
    selected_xpus = [x.strip() for x in selected_xpus.split(',')]
    return selected_xpus


def start_local_trainers(
    cluster,
    pod,
    training_script,
    training_script_args,
    eager_mode=True,
    log_dir=None,
):
    current_env = copy.copy(os.environ.copy())
    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for t in pod.trainers:
        proc_env = {
            "PADDLE_DISTRI_BACKEND": "bkcl",
            "FLAGS_selected_xpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints()),
        }

        current_env.update(proc_env)

        print(f"trainer proc env:{current_env}")

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            cmd = "python -m coverage run --branch -p " + training_script
        else:
            cmd = "python -u " + training_script

        print(f"start trainer proc:{cmd} env:{proc_env}")

        fn = None

        proc = subprocess.Popen(cmd.split(" "), env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


class TestMultipleXpus(unittest.TestCase):
    def run_mnist_2xpu(self, target_file_name, eager_mode=True):
        if (
            not fluid.core.is_compiled_with_xpu()
            or fluid.core.get_xpu_device_count() == 0
        ):
            return

        selected_xpus = get_xpus('0,1')
        paddle.set_device("xpu")
        cluster = None
        pod = None

        cluster, pod = get_cluster_from_args(selected_xpus)

        procs = start_local_trainers(
            cluster,
            pod,
            eager_mode=eager_mode,
            training_script=target_file_name,
            training_script_args=[],
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())

            if not alive:
                print(f"Local procs complete, POD info:{pod}")
                break
            time.sleep(3)


class TestDataParallelWithPyLayer(TestMultipleXpus):
    def test_parallel_dygraph_dataparallel_with_pylayer(self):
        self.run_mnist_2xpu('parallel_dygraph_dataparallel_with_pylayer.py')


class TestGradientCheckInEagerMode(TestMultipleXpus):
    def test_multiple_xpus_dynamic(self):
        self.run_mnist_2xpu('parallel_dygraph_gradient_check_in_eager_mode.py')


if __name__ == "__main__":
    os.environ["BKCL_PCIE_RING"] = "1"
    os.environ["BKCL_CCIX_RING"] = "0"
    unittest.main()
