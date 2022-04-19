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
import paddle
import paddle.fluid as fluid
import copy
import os
import subprocess

from paddle.distributed.utils import find_free_ports, watch_local_trainers, get_cluster, TrainerProc
from paddle.fluid.framework import _test_eager_guard


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


def start_local_trainers_cpu(trainer_endpoints,
                             training_script,
                             training_script_args,
                             log_dir=None):
    current_env = copy.copy(os.environ.copy())
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    n_rank = len(trainer_endpoints)
    print(trainer_endpoints)
    for rank_id, endpoint in enumerate(trainer_endpoints):
        proc_env = {
            "PADDLE_DISTRI_BACKEND": "gloo",
            "PADDLE_TRAINER_ID": "%d" % rank_id,
            "PADDLE_CURRENT_ENDPOINT": "%s" % endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % n_rank,
            "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints)
        }

        current_env.update(proc_env)

        print("trainer proc env:{}".format(current_env))

        assert os.getenv('WITH_COVERAGE',
                         'OFF') == 'OFF', "Gloo don't support WITH_COVERAGE."
        cmd = "python -u " + training_script

        print("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None

        proc = subprocess.Popen(cmd.split(" "), env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = rank_id
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


def start_local_trainers(cluster,
                         pod,
                         training_script,
                         training_script_args,
                         eager_mode=True,
                         log_dir=None):
    current_env = copy.copy(os.environ.copy())
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for t in pod.trainers:
        proc_env = {
            "FLAGS_selected_gpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints())
        }

        if not eager_mode:
            proc_env["FLAGS_enable_eager_mode"] = "%d" % 0

        current_env.update(proc_env)

        print("trainer proc env:{}".format(current_env))

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            cmd = "python -m coverage run --branch -p " + training_script
        else:
            cmd = "python -u " + training_script

        print("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None

        proc = subprocess.Popen(cmd.split(" "), env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


class TestMultipleGpus(unittest.TestCase):
    def run_mnist_2gpu(self, target_file_name, eager_mode=True):
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
            eager_mode=eager_mode,
            training_script=target_file_name,
            training_script_args=[])

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())

            if not alive:
                print("Local procs complete, POD info:{}".format(pod))
                break
            time.sleep(3)


class TestMultipleWithGloo(unittest.TestCase):
    def run_mnist_2cpu(self, target_file_name):

        cluster, pod = get_cluster_from_args(
            [0, 1])  #tmp use. for getting trainer_nranks()

        procs = start_local_trainers_cpu(
            cluster.trainers_endpoints(),
            training_script=target_file_name,
            training_script_args=[])

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_nranks())

            if not alive:
                print("Local procs complete, POD info:{}".format(pod))
                break
            time.sleep(3)


class TestDataParallelGradientCheck(TestMultipleGpus):
    def test_multiple_gpus_dynamic(self):
        self.run_mnist_2gpu('parallel_dygraph_gradient_check.py')


class TestDataParallelWithPyLayer(TestMultipleGpus):
    def test_parallel_dygraph_dataparallel_with_pylayer(self):
        self.run_mnist_2gpu('parallel_dygraph_dataparallel_with_pylayer.py')
        self.run_mnist_2gpu(
            'parallel_dygraph_dataparallel_with_pylayer.py', eager_mode=False)


class TestGradientCheckInEagerMode(TestMultipleGpus):
    def test_multiple_gpus_dynamic(self):
        self.run_mnist_2gpu('parallel_dygraph_gradient_check_in_eager_mode.py')


if __name__ == "__main__":
    unittest.main()
