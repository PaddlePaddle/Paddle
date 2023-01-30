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

<<<<<<< HEAD
import copy
import os
import subprocess
import tempfile
import time
import unittest


def start_local_trainers(
    cluster,
    pod,
    training_script,
    training_script_args,
    eager_mode=True,
    log_dir=None,
):
    from paddle.distributed.utils.launch_utils import (  # noqa: F401
        TrainerProc,
        find_free_ports,
        get_cluster,
        watch_local_trainers,
    )

    current_env = copy.copy(os.environ.copy())
    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.
=======
import unittest
import os
import copy
import subprocess
import time
import tempfile


def start_local_trainers(cluster,
                         pod,
                         training_script,
                         training_script_args,
                         eager_mode=True,
                         log_dir=None):
    from paddle.distributed.utils.launch_utils import find_free_ports, watch_local_trainers, get_cluster, TrainerProc  # noqa: F401

    current_env = copy.copy(os.environ.copy())
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []

    os.system("rm -rf log && mkdir -p log")
    for idx, t in enumerate(pod.trainers):
        proc_env = {
<<<<<<< HEAD
            "FLAGS_selected_custom_cpus": "%s"
            % ",".join([str(g) for g in t.gpus]),
=======
            "FLAGS_selected_custom_cpus":
            "%s" % ",".join([str(g) for g in t.gpus]),
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints()),
            "PADDLE_DISTRI_CUSTOM_DEVICE_TYPE": "custom_cpu",
        }

<<<<<<< HEAD
=======
        if not eager_mode:
            proc_env["FLAGS_enable_eager_mode"] = "%d" % 0

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        current_env.update(proc_env)

        print("trainer proc env:{}".format(current_env))

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            cmd = "python -m coverage run --branch -p " + training_script
        else:
            cmd = "python -u " + training_script

        print("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = open("workerlog.%d" % idx, "a")
<<<<<<< HEAD
        proc = subprocess.Popen(
            cmd.split(" "), env=current_env, stdout=fn, stderr=fn
        )
=======
        proc = subprocess.Popen(cmd.split(" "),
                                env=current_env,
                                stdout=fn,
                                stderr=fn)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


def get_cluster_from_args(selected_gpus):
<<<<<<< HEAD
    from paddle.distributed.utils.launch_utils import (  # noqa: F401
        TrainerProc,
        find_free_ports,
        get_cluster,
        watch_local_trainers,
    )
=======
    from paddle.distributed.utils.launch_utils import find_free_ports, watch_local_trainers, get_cluster, TrainerProc  # noqa: F401
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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


class TestMultipleCustomCPU(unittest.TestCase):
<<<<<<< HEAD
    def run_mnist_2custom_cpu(self, target_file_name, eager_mode=True):
        from paddle.distributed.utils.launch_utils import (  # noqa: F401
            TrainerProc,
            find_free_ports,
            get_cluster,
            watch_local_trainers,
        )
=======

    def run_mnist_2custom_cpu(self, target_file_name, eager_mode=True):
        from paddle.distributed.utils.launch_utils import find_free_ports, watch_local_trainers, get_cluster, TrainerProc  # noqa: F401
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        selected_devices = [0, 1]
        cluster = None
        pod = None

        cluster, pod = get_cluster_from_args(selected_devices)

<<<<<<< HEAD
        procs = start_local_trainers(
            cluster,
            pod,
            eager_mode=eager_mode,
            training_script=target_file_name,
            training_script_args=[],
        )
=======
        procs = start_local_trainers(cluster,
                                     pod,
                                     eager_mode=eager_mode,
                                     training_script=target_file_name,
                                     training_script_args=[])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())

            if not alive:
                print("Local procs complete, POD info:{}".format(pod))
                break
            time.sleep(3)


class TestProcessGroup(TestMultipleCustomCPU):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        cmd = 'cd {} \
            && git clone {} \
            && cd PaddleCustomDevice \
            && git fetch origin \
            && git checkout {} -b dev \
            && cd backends/custom_cpu \
            && mkdir build && cd build && cmake .. && make -j8'.format(
<<<<<<< HEAD
            self.temp_dir.name, os.getenv('PLUGIN_URL'), os.getenv('PLUGIN_TAG')
        )
=======
            self.temp_dir.name, os.getenv('PLUGIN_URL'),
            os.getenv('PLUGIN_TAG'))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        os.system(cmd)

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
<<<<<<< HEAD
            cur_dir,
            '{}/PaddleCustomDevice/backends/custom_cpu/build'.format(
                self.temp_dir.name
            ),
        )
=======
            cur_dir, '{}/PaddleCustomDevice/backends/custom_cpu/build'.format(
                self.temp_dir.name))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        os.environ['FLAGS_selected_custom_cpus'] = '0,1'
        os.environ['CUSTOM_CPU_VISIBLE_DEVICES'] = '0,1'
        os.environ['PADDLE_XCCL_BACKEND'] = 'custom_cpu'

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_group_xccl(self):
<<<<<<< HEAD
        from paddle.distributed.utils.launch_utils import (  # noqa: F401
            TrainerProc,
            find_free_ports,
            get_cluster,
            watch_local_trainers,
        )
=======
        from paddle.distributed.utils.launch_utils import find_free_ports, watch_local_trainers, get_cluster, TrainerProc  # noqa: F401
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.run_mnist_2custom_cpu('process_group_xccl.py')


if __name__ == "__main__":
    unittest.main()
