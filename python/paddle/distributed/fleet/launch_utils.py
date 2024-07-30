# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import json
import logging
import multiprocessing
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import closing

import paddle.utils.cpp_extension.extension_utils as utils
from paddle import framework
from paddle.utils import strtobool

logger = logging.getLogger("root")
logger.propagate = False


class DistributeMode:
    """
    There are various mode for fleetrun, each of them is designed for different model.
    """

    COLLECTIVE = 0
    PS = 1
    PS_HETER = 2


class DeviceMode:
    """
    Training devices type
    """

    UNKNOWN = -1
    CPU = 0
    GPU = 1
    KUNLUN = 2
    XPU = 2


class Cluster:
    def __init__(self, hdfs):
        self.job_server = None
        self.pods = []
        self.hdfs = None
        self.job_stage_flag = None

    def __str__(self):
        return f"job_server:{self.job_server} pods:{[str(pod) for pod in self.pods]} job_stage_flag:{self.job_stage_flag} hdfs:{self.hdfs}"

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return False

        for a, b in zip(self.pods, cluster.pods):
            if a != b:
                return False

        if self.job_stage_flag != cluster.job_stage_flag:
            return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(self, cluster):
        self.pods = copy.copy(cluster.pods)

    def trainers_nranks(self):
        return len(self.trainers_endpoints())

    def pods_nranks(self):
        return len(self.pods)

    def trainers_endpoints(self):
        r = []
        for pod in self.pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def world_device_ids(self):
        r = []
        for pod in self.pods:
            for t in pod.trainers:
                str_accelerators = [str(acc) for acc in t.accelerators]
                r.append(str_accelerators)
        return r

    def pods_endpoints(self):
        r = []
        for pod in self.pods:
            ep = f"{pod.addr}:{pod.port}"
            assert (
                pod.port is not None and pod.addr is not None
            ), f"{ep} not a valid endpoint"
            r.append(ep)
        return r

    def get_pod_by_id(self, pod_id):
        for pod in self.pods:
            if str(pod_id) == str(pod.id):
                return pod

        return None


class JobServer:
    def __init__(self):
        self.endpoint = None

    def __str__(self):
        return f"{self.endpoint}"

    def __eq__(self, j):
        return self.endpoint == j.endpoint

    def __ne__(self, j):
        return not self == j


class Trainer:
    def __init__(self):
        self.accelerators = []
        self.endpoint = None
        self.rank = None
        self.stage = None

    def __str__(self):
        return f"accelerator:{self.accelerators} endpoint:{self.endpoint} rank:{self.rank}"

    def __eq__(self, t):
        if len(self.accelerators) != len(t.accelerators):
            return False

        if self.endpoint != t.endpoint or self.rank != t.rank:
            return False

        for a, b in zip(self.accelerators, t.accelerators):
            if a != b:
                return False

        return True

    def __ne__(self, t):
        return not self == t

    def rank(self):
        return self.rank


class Pod:
    def __init__(self):
        self.rank = None
        self.id = None
        self.addr = None
        self.port = None
        self.trainers = []
        self.servers = []
        self.workers = []
        self.coordinators = []
        self.heter_workers = []
        self.accelerators = []
        self.device_mode = None

    def __str__(self):
        return f"rank:{self.rank} id:{self.id} addr:{self.addr} port:{self.port} visible_accelerator:{self.accelerators} trainers:{[str(t) for t in self.trainers]} servers:{[str(s) for s in self.servers]} \
            workers:{[str(w) for w in self.workers]} heter_workers:{[str(h) for h in self.heter_workers]} coordinators:{[str(c) for c in self.coordinators]}"

    def __eq__(self, pod):
        if (
            self.rank != pod.rank
            or self.id != pod.id
            or self.addr != pod.addr
            or self.port != pod.port
        ):
            logger.debug(f"pod {self} != {pod}")
            return False

        if len(self.trainers) != len(pod.trainers):
            logger.debug(f"trainers {self.trainers} != {pod.trainers}")
            return False

        for i in range(len(self.trainers)):
            if self.trainers[i] != pod.trainers[i]:
                logger.debug(f"trainer {self.trainers[i]} != {pod.trainers[i]}")
                return False

        if len(self.servers) != len(pod.servers):
            logger.debug(f"servers {self.servers} != {pod.servers}")
            return False

        for i in range(len(self.servers)):
            if self.servers[i] != pod.servers[i]:
                logger.debug(f"servers {self.servers[i]} != {pod.servers[i]}")
                return False

        if len(self.workers) != len(pod.workers):
            logger.debug(f"workers {self.workers} != {pod.workers}")
            return False

        for i in range(len(self.workers)):
            if self.workers[i] != pod.workers[i]:
                logger.debug(f"workers {self.workers[i]} != {pod.workers[i]}")
                return False

        return True

    def __ne__(self, pod):
        return not self == pod

    def parse_response(self, res_pods):
        pass

    def rank(self):
        return self.rank

    def get_visible_accelerators(self):
        r = ""
        for g in self.accelerators:
            r += f"{g},"

        assert r != "", f"this pod {self} can't see any accelerators"

        r = r[:-1]
        return r


def get_logger(log_level=20, name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
    )
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


def get_cluster(
    node_ips, node_ip, trainer_endpoints, device_mode, devices_per_proc
):
    assert type(trainer_endpoints) is list, "trainer_endpoints must be list"
    cluster = Cluster(hdfs=None)
    trainer_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        pod.device_mode = device_mode

        cur_node_endpoints = trainer_endpoints[node_rank]
        # when use paddlecloud, endpoints may > devices_per_proc(user_defined)
        assert len(cur_node_endpoints) >= len(
            devices_per_proc
        ), "current trainer_endpoints size should be greater equal than acclerators size."
        for i in range(len(devices_per_proc)):
            trainer = Trainer()
            if device_mode == DeviceMode.GPU:
                if isinstance(devices_per_proc[i], (list, tuple)):
                    trainer.accelerators.extend(devices_per_proc[i])
                    pod.accelerators.extend(devices_per_proc[i])
                else:
                    trainer.accelerators.append(devices_per_proc[i])
                    pod.accelerators.append(devices_per_proc[i])
            elif device_mode == DeviceMode.XPU:
                if isinstance(devices_per_proc[i], (list, tuple)):
                    trainer.accelerators.extend(devices_per_proc[i])
                else:
                    trainer.accelerators.append(devices_per_proc[i])
            trainer.endpoint = f"{cur_node_endpoints[i]}"
            trainer.rank = trainer_rank
            trainer_rank += 1

            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def terminate_local_procs(procs):
    # try to terminate process by group, this happened in multiprocess scenario in user process
    if os.name != 'nt':
        for p in procs:
            if p.proc.poll() is None:
                os.killpg(os.getpgid(p.proc.pid), signal.SIGTERM)
                if p.log_fn:
                    p.log_fn.close()
                logger.info(f"terminate process group gid:{p.proc.pid}")

        time.sleep(1)

    for p in procs:
        if p.proc.poll() is None:
            p.proc.terminate()
            if p.log_fn:
                p.log_fn.close()
            logger.debug(f"terminate process id:{p.proc.pid}")

    # wait all process terminated
    time.sleep(3)
    for step in range(0, 50):
        alive = False
        for p in procs:
            if p.proc.poll() is None:  # not terminate
                os.kill(p.proc.pid, signal.SIGKILL)
                alive = True

        if not alive:
            logger.info("terminate all the procs")
            return

        time.sleep(3)

    logger.fatal("can't kill all process and exit")
    sys.exit(1)


def get_host_name_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        return None


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Examples:
        .. code-block:: python

            >>> import argparse
            >>> from paddle.distributed.fleet.launch_utils import add_arguments
            >>> parser = argparse.ArgumentParser()
            >>> add_arguments("name", str, "Jonh", "User name.", parser)
            >>> args = parser.parse_args()

    """
    type = strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs,
    )


def find_free_ports(num):
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            # Note(wangxi): Close the connection with a TCP RST instead
            # of a TCP FIN, to avoid time_wait state.
            s.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0)
            )
            s.bind(('', 0))
            return s.getsockname()[1]

    port_set = set()
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 400:
            print(
                "can't find available port and use the specified static port now!"
            )
            return None

    return None


def get_ports(num, offset):
    if os.environ.get('FLAGS_START_PORT') is None:
        ports = find_free_ports(num)
        if ports is not None:
            ports = list(ports)
    else:
        start_port = int(os.environ.get('FLAGS_START_PORT'))
        ports = range(start_port + offset, start_port + offset + num, 1)
    return ports


def pretty_print_envs(envs, header=None):
    spacing = 2
    max_k = 40
    max_v = 45

    for k, v in envs.items():
        max_k = max(max_k, len(k))

    h_format = "    " + "|{{:>{}s}}{}{{:^{}s}}|\n".format(
        max_k, " " * spacing, max_v
    )
    l_format = "    " + f"|{{:>{max_k}s}}{{}}{{:^{max_v}s}}|\n"
    length = max_k + max_v + spacing

    border = "    +" + "".join(["="] * length) + "+"
    line = "    +" + "".join(["-"] * length) + "+"

    draws = ""
    draws += border + "\n"

    if header:
        draws += h_format.format(header[0], header[1])
    else:
        draws += h_format.format("fleetrun Distributed Envs", "Value")

    draws += line + "\n"

    for k, v in envs.items():
        if isinstance(v, str) and len(v) >= max_v:
            str_v = "... " + v[-41:]
        else:
            str_v = v

        draws += l_format.format(k, " " * spacing, str(str_v))

    draws += border

    _str = f"\n{draws}\n"
    return _str


class TrainerProc:
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.log_offset = None
        self.rank = None
        self.local_rank = None
        self.cmd = None


_run_with_coverage = False


def run_with_coverage(*args):
    global _run_with_coverage
    assert len(args) <= 1, f"len(args) {len(args)} should <= 1"
    if len(args) == 1:
        assert isinstance(args[0], bool)
        _run_with_coverage = args[0]
    return _run_with_coverage


def start_local_trainers(
    cluster, pod, training_script, training_script_args, log_dir=None, envs=None
):
    if envs is None:
        current_env = copy.copy(os.environ.copy())
    else:
        current_env = copy.copy(envs)

    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    ids = cluster.world_device_ids()
    res = [':'.join(ele) for ele in ids]
    procs = []
    for idx, t in enumerate(pod.trainers):
        proc_env = {
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": f"{t.endpoint}",
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints()),
            "PADDLE_RANK_IN_NODE": str(idx),
            "PADDLE_LOCAL_DEVICE_IDS": ",".join(
                [str(acc) for acc in t.accelerators]
            ),
            "PADDLE_WORLD_DEVICE_IDS": ",".join(res),
        }

        # The following three environnement variables are used for auto mapping
        if current_env.get("PADDLE_CLUSTER_TOPO_PATH", None) is not None:
            proc_env["PADDLE_CLUSTER_TOPO_PATH"] = current_env[
                "PADDLE_CLUSTER_TOPO_PATH"
            ]
        if current_env.get("PADDLE_RANK_MAPPING_PATH", None) is not None:
            proc_env["PADDLE_RANK_MAPPING_PATH"] = current_env[
                "PADDLE_RANK_MAPPING_PATH"
            ]
        if current_env.get("PADDLE_ENABLE_AUTO_MAPPING", None) is not None:
            proc_env["PADDLE_ENABLE_AUTO_MAPPING"] = current_env[
                "PADDLE_ENABLE_AUTO_MAPPING"
            ]

        if len(t.accelerators) > 0 and pod.device_mode == DeviceMode.GPU:
            proc_env["FLAGS_selected_gpus"] = "{}".format(
                ",".join([str(g) for g in t.accelerators])
            )

        if len(t.accelerators) > 0:
            proc_env["FLAGS_selected_accelerators"] = "{}".format(
                ",".join([str(g) for g in t.accelerators])
            )
        # to do: same code style in future
        if framework.core.is_compiled_with_xpu() and len(t.accelerators) > 0:
            proc_env["FLAGS_selected_xpus"] = "{}".format(
                ",".join([str(g) for g in t.accelerators])
            )

        current_env.update(proc_env)

        coverage_args = []
        if (
            run_with_coverage()
            or os.environ.get("WITH_COVERAGE", "OFF") == "ON"
        ):
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        cmd = (
            [sys.executable, "-u"]
            + coverage_args
            + [training_script]
            + training_script_args
        )

        logger.debug(f"start trainer proc{cmd}  env:{current_env}")

        if idx == 0:
            logger.info(
                "Local start {} processes. First process distributed "
                "environment info (Only For Debug): {}".format(
                    len(pod.trainers),
                    pretty_print_envs(proc_env, ("Distributed Envs", "Value")),
                )
            )
            logger.info(
                "details about PADDLE_TRAINER_ENDPOINTS can be found in "
                f"{log_dir}/endpoints.log, and detail running logs maybe found in "
                f"{log_dir}/workerlog.0"
            )
        fn = None
        pre_fn = None if os.name == 'nt' else os.setsid
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            if os.path.exists(f"{log_dir}/endpoints.log"):
                os.remove(f"{log_dir}/endpoints.log")
            with open(f"{log_dir}/endpoints.log", "w") as f:
                f.write("PADDLE_TRAINER_ENDPOINTS: \n")
                f.write("\n".join(cluster.trainers_endpoints()))
            if (
                current_env.get("PADDLE_ENABLE_AUTO_MAPPING") is not None
                and current_env.get("PADDLE_NEED_RANK_MAPPING").lower()
                == "true"
            ):
                fn = open("%s/prelaunchlog.%d" % (log_dir, idx), "a")
            else:
                fn = open("%s/workerlog.%d" % (log_dir, idx), "a")
            proc = subprocess.Popen(
                cmd, env=current_env, stdout=fn, stderr=fn, preexec_fn=pre_fn
            )
        else:
            proc = subprocess.Popen(cmd, env=current_env, preexec_fn=pre_fn)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.local_rank = idx
        tp.log_fn = fn
        tp.log_offset = fn.tell() if fn else None
        tp.cmd = cmd

        procs.append(tp)

    return procs


def pull_worker_log(tp):
    if tp.log_fn:
        with open(tp.log_fn.name, 'r') as fin:
            fin.seek(tp.log_offset, 0)
            for line in fin:
                try:
                    sys.stdout.write(line)
                except UnicodeEncodeError:
                    sys.stdout.write(
                        'UnicodeEncodeError occurs at this line. '
                        f'Please refer to the original log file "{tp.log_fn.name}"\n'
                    )
            tp.log_offset = fin.tell()


def watch_local_trainers(procs, nranks):
    try:
        error = False
        error_rank = []
        # wait all process finish or one error
        alive = False
        for p in procs:
            if p.log_fn and p.local_rank == 0:
                pull_worker_log(p)

            ret = p.proc.poll()
            if ret is None:
                alive = True
            elif ret != 0:
                error = True
                error_rank.append(p.rank)

        if error:
            terminate_local_procs(procs)
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exit")
        terminate_local_procs(procs)
        return
    except SystemExit:
        logger.error(
            f"ABORT!!! Out of all {nranks} trainers, the trainer process with rank={error_rank} was aborted. Please check its log."
        )
        terminate_local_procs(procs)
        raise
    except:
        logger.error(
            f"ABORT!!! Out of all {nranks} trainers, the trainer process with rank={error_rank} was aborted. Please check its log."
        )
        terminate_local_procs(procs)
        return

    return alive


def get_gpus(gpus):
    if gpus is None:
        gpus_num = framework.core.get_cuda_device_count()
        res_gpus = [str(x) for x in range(0, gpus_num)]
    else:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or cuda_visible_devices == "":
            res_gpus = [x.strip() for x in gpus.split(',')]
        else:
            # change gpus into relative values
            # e.g. CUDA_VISIBLE_DEVICES=4,5,6,7; args.gpus=4,5,6,7;
            # therefore gpus=0,1,2,3
            cuda_visible_devices_list = cuda_visible_devices.split(',')
            for x in gpus.split(','):
                assert x in cuda_visible_devices_list, (
                    "Can't find "
                    f"your gpus {x} in CUDA_VISIBLE_DEVICES[{cuda_visible_devices}]."
                )
            res_gpus = [
                cuda_visible_devices_list.index(x.strip())
                for x in gpus.split(',')
            ]
            logger.info(
                f"Change selected_gpus into relative values. --ips:{gpus} "
                f"will change into relative_ips:{res_gpus} according to your "
                f"CUDA_VISIBLE_DEVICES:{cuda_visible_devices_list}"
            )

    return res_gpus


def get_xpus(xpus):
    if xpus is None:
        xpus_num = framework.core.get_xpu_device_count()
        res_xpus = [str(x) for x in range(0, xpus_num)]
    else:
        xpu_visible_devices = os.getenv("XPU_VISIBLE_DEVICES")
        if xpu_visible_devices is None or xpu_visible_devices == "":
            res_xpus = [x.strip() for x in xpus.split(',')]
        else:
            # change xpus into relative values
            # e.g. XPU_VISIBLE_DEVICES=4,5,6,7; args.xpus=4,5,6,7;
            # therefore xpus=0,1,2,3
            xpu_visible_devices_list = xpu_visible_devices.split(',')
            for x in xpus.split(','):
                assert x in xpu_visible_devices_list, (
                    "Can't find "
                    f"your xpus {x} in XPU_VISIBLE_DEVICES[{xpu_visible_devices}]."
                )
            res_xpus = [
                xpu_visible_devices_list.index(x.strip())
                for x in xpus.split(',')
            ]
            logger.info(
                f"Change selected_xpus into relative values. --ips:{xpus} "
                f"will change into relative_ips:{res_xpus} according to your "
                f"XPU_VISIBLE_DEVICES:{xpu_visible_devices_list}"
            )

    return res_xpus


def get_device_mode(backend):
    if backend == 'heter':
        if (
            framework.core.is_compiled_with_cuda()
            and framework.core.get_cuda_device_count() > 0
        ):
            print("launch train in heter mode with GPU device.")
            return DeviceMode.GPU
        if (
            framework.core.is_compiled_with_xpu()
            and framework.core.get_xpu_device_count() > 0
        ):
            print("launch train in heter mode with XPU device.")
            return DeviceMode.XPU

    if backend == 'nccl' and framework.core.get_cuda_device_count() > 0:
        print("launch train in GPU mode!")
        return DeviceMode.GPU

    if backend == 'bkcl' and framework.core.get_xpu_device_count() > 0:
        print("launch train in XPU mode")
        return DeviceMode.XPU

    if backend == 'gloo':
        print("launch train in CPU mode")
        return DeviceMode.CPU

    raise RuntimeError("Don't supported devices")


def get_device_proc_info(args):
    # device_mode
    device_mode = get_device_mode(args.backend)

    # devices
    devices_per_proc = []
    if device_mode == DeviceMode.GPU:
        gpus = get_gpus(args.gpus)
        if args.nproc_per_node is not None:
            assert (
                len(gpus) % int(args.nproc_per_node)
            ) == 0, f"gpus' number:{len(gpus)} mod args.nproc_per_node:{args.nproc_per_node} must == 0"

            n = int(len(gpus) / int(args.nproc_per_node))
            devices_per_proc = [gpus[i : i + n] for i in range(0, len(gpus), n)]
        else:
            devices_per_proc = gpus
    elif device_mode == DeviceMode.XPU:
        xpus = get_xpus(args.xpus)
        if args.nproc_per_node is not None:
            assert (
                len(xpus) % int(args.nproc_per_node)
            ) == 0, f"xpus' number:{len(xpus)} mod args.nproc_per_node:{args.nproc_per_node} must == 0"

            n = int(len(xpus) / int(args.nproc_per_node))
            devices_per_proc = [xpus[i : i + n] for i in range(0, len(xpus), n)]
        else:
            devices_per_proc = xpus
    elif device_mode == DeviceMode.CPU:
        if hasattr(args, "paddle_cpuonly") and args.nproc_per_node is None:
            # NOTE (xiongkun03) set it to cpu core number
            args.nproc_per_node = multiprocessing.cpu_count()
        if args.nproc_per_node is None:
            devices_per_proc = [0]
        else:
            devices_per_proc = list(range(0, args.nproc_per_node))
    else:
        raise AssertionError(
            f"Can't support device_mode:{device_mode}, support only cpu|gpu|xpu now."
        )

    return (device_mode, devices_per_proc)


def direct_start(args):
    # run ps-cpu mode on paddlecloud, using given envs
    cmd = [
        sys.executable,
        "-u",
        args.training_script,
    ] + args.training_script_args
    proc = subprocess.Popen(cmd)
    proc.wait()


def get_custom_endpoints(origin_endpoints, offset=0):
    """
    origin_endpoint: ip:port
    user_define_endpoint: ip:(port+offset)
    """
    assert origin_endpoints is not None
    paddle_user_define_endpoints_list = []
    for ip_port in origin_endpoints.split(","):
        ip = ip_port.split(":")[0]
        port = ip_port.split(":")[1]
        new_port = int(port) + offset
        paddle_user_define_endpoints_list.append(":".join((ip, str(new_port))))
    paddle_user_define_endpoints = ",".join(paddle_user_define_endpoints_list)
    return paddle_user_define_endpoints


# def cloud_ps_heter_env_set(args):
#    environs = {}
#
#    paddle_trainer_endpoints = os.getenv("TRAINER_IP_PORT_LIST", "")
#    assert paddle_trainer_endpoints != None
#
#    paddle_pserver_endpoints = os.getenv("PSERVER_IP_PORT_LIST", "")
#    assert paddle_pserver_endpoints != None
#
#    # hard code for paddlecloud custom-framework
#    available_ports = os.getenv("TRAINER_PORTS", "").split(",")
#    assert len(
#        available_ports
#    ) >= 2, "set paddle_ports_num >= 2 in config.ini for paddlecloud job submit"
#
#    # hard code for paddlecloud custom-framework
#    trainers_num = len(paddle_pserver_endpoints.split(","))
#    assert trainers_num != 0
#    environs["PADDLE_TRAINERS_NUM"] = trainers_num
#    environs["TRAINERS_NUM"] = trainers_num
#
#    # hard code for paddlecloud custom-framework
#    environs["PADDLE_HETER_TRAINER_IP_PORT_LIST"] = paddle_trainer_endpoints
#    environs["PADDLE_PSERVERS_IP_PORT_LIST"] = paddle_pserver_endpoints
#    environs["PADDLE_TRAINER_ENDPOINTS"] = get_custom_endpoints(
#        paddle_pserver_endpoints, 1)
#    heter_worker_num = len(paddle_trainer_endpoints.split(","))
#    if (args.heter_worker_num != None) and (
#            heter_worker_num != args.heter_worker_num):
#        warnings.warn(
#            "Your fleetrun setting: heter_worker_num is {}, but we find {} device can be used, this setting has been changed.".
#            format(args.heter_worker_num, heter_worker_num))
#        args.heter_worker_num = heter_worker_num
#
#    for k, v in environs.items():
#        os.environ[k] = str(v)
#    logger.info("Set heter parameter server env: {}".format(
#        pretty_print_envs(environs)))


def get_mapped_cluster_without_rank_mapping(
    node_ips, node_ip, trainer_endpoints, device_mode, node_ranks
):
    assert type(trainer_endpoints) is list, "trainer_endpoints must be list"
    assert (
        device_mode == DeviceMode.GPU
    ), "Only support get mapped cluster for gpu now."
    cluster = Cluster(hdfs=None)
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        pod.device_mode = device_mode
        cur_node_endpoints = trainer_endpoints[node_rank]

        # choose rank from global mapped ranks and set it to the trainer.
        ranks_per_node = node_ranks[node_rank]
        assert len(ranks_per_node) == 1
        for i in range(len(ranks_per_node)):
            trainer = Trainer()
            trainer.endpoint = f"{cur_node_endpoints[i]}"
            trainer.rank = ranks_per_node[i]
            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def get_mapped_cluster_from_args_without_rank_mapping(args, device_mode):
    assert (
        device_mode == DeviceMode.GPU
    ), "Only support get mapped cluster for gpu now."
    gpus_num = framework.core.get_cuda_device_count()

    # parse ip-ranks json file
    cluster_topo = None
    with open(args.cluster_topo_path, "r") as json_file:
        cluster_topo = json.load(json_file)

    node_ips = []
    node_ranks = []
    for idx, cur_cluster_topo in enumerate(cluster_topo["machines"]):
        node_ips.append(cur_cluster_topo['addr'])
        node_ranks.append([idx])

    if len(node_ips) == 1:
        node_ip = node_ips[0]
    else:
        if args.host:
            node_ip = args.host
        else:
            _, node_ip = get_host_name_ip()

    assert (
        node_ip in node_ips
    ), f"Can't find your local ip {{{node_ip}}} in node_ips: {{{node_ips}}}"
    node_rank = node_ips.index(node_ip)

    assert len(node_ranks) == len(
        node_ips
    ), "ranks length should be equal to ips length."

    logger.debug(
        f"parsed from args: node_ips:{node_ips} node_ip:{node_ip} "
        f"node_rank:{node_rank} node_ranks:{node_ranks[node_rank]}"
    )

    # NOTE: there are different number of global mapped ranks on each node.
    free_ports = []
    trainer_endpoints = []
    for ip in node_ips:
        node_rank = node_ips.index(ip)
        if os.environ.get('PADDLE_PORT') is not None:
            start_port = int(os.getenv("PADDLE_PORT", ""))
            free_ports = list(
                range(start_port, start_port + len(node_ranks[node_rank]))
            )
        elif os.environ.get('FLAGS_START_PORT') is not None:
            start_port = int(os.environ.get('FLAGS_START_PORT'))
            free_ports = list(
                range(start_port, start_port + len(node_ranks[node_rank]))
            )
        else:
            free_ports = find_free_ports(len(node_ranks[node_rank]))
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])

    return get_mapped_cluster_without_rank_mapping(
        node_ips, node_ip, trainer_endpoints, device_mode, node_ranks
    )


def get_mapped_cluster_with_rank_mapping(
    node_ips,
    node_ip,
    trainer_endpoints,
    device_mode,
    node_ranks,
    node_rank_mappings,
):
    assert type(trainer_endpoints) is list, "trainer_endpoints must be list"
    assert (
        device_mode == DeviceMode.GPU
    ), "Only support get mapped cluster for gpu now."

    def get_relative_gpu_id(gpu_id):
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or cuda_visible_devices == "":
            return gpu_id
        else:
            cuda_visible_devices_list = cuda_visible_devices.split(',')
            relative_id = cuda_visible_devices_list.index(str(gpu_id))
            logger.info(
                f"Change gpu id from {gpu_id} to {relative_id} based on CUDA_VISIBLE_DEVICES {cuda_visible_devices_list}"
            )
            return relative_id

    cluster = Cluster(hdfs=None)
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        pod.device_mode = device_mode
        cur_node_endpoints = trainer_endpoints[node_rank]

        # choose rank from global mapped ranks and set it to the trainer.
        ranks_per_node = node_ranks[node_rank]
        cur_node_rank_mapping = node_rank_mappings[node_rank]
        for i in range(len(ranks_per_node)):
            trainer = Trainer()
            local_device_ids = cur_node_rank_mapping["ranks"][
                str(ranks_per_node[i])
            ]
            assert (
                len(local_device_ids) == 1
            ), "Only support one process to one device mapping"
            trainer.accelerators.append(
                get_relative_gpu_id(local_device_ids[0])
            )
            trainer.endpoint = f"{cur_node_endpoints[i]}"
            trainer.rank = ranks_per_node[i]
            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def get_mapped_cluster_from_args_with_rank_mapping(args, device_mode):
    assert (
        device_mode == DeviceMode.GPU
    ), "Only support get mapped cluster for gpu now."
    gpus_num = framework.core.get_cuda_device_count()

    # parse ip-ranks json file
    rank_mapping_path = args.rank_mapping_path or os.getenv(
        "PADDLE_RANK_MAPPING_PATH"
    )
    rank_mapping = None
    with open(rank_mapping_path, "r") as json_file:
        rank_mapping = json.load(json_file)
    # reset PADDLE_RANK_MAPPING_PATH env
    os.environ["PADDLE_RANK_MAPPING_PATH"] = ""

    node_ips = []
    node_ranks = []
    node_rank_mappings = []
    for cur_rank_mapping in rank_mapping:
        node_ips.append(cur_rank_mapping['addr'])
        cur_node_rank_list = [
            int(i) for i in list(cur_rank_mapping['ranks'].keys())
        ]
        cur_node_rank_list.sort()
        node_ranks.append(cur_node_rank_list)
        node_rank_mappings.append(cur_rank_mapping)

    if len(node_ips) == 1:
        node_ip = node_ips[0]
    else:
        if args.host:
            node_ip = args.host
        else:
            _, node_ip = get_host_name_ip()

    assert (
        node_ip in node_ips
    ), f"Can't find your local ip {{{node_ip}}} in node_ips: {{{node_ips}}}"
    node_rank = node_ips.index(node_ip)

    assert (
        len(node_ranks[node_rank]) <= gpus_num
    ), "number of ranks mapped to one node should not exceed the available ones."
    assert len(node_ranks) == len(
        node_ips
    ), "ranks length should be equal to ips length."

    logger.debug(
        f"parsed from args: node_ips:{node_ips} node_ip:{node_ip} "
        f"node_rank:{node_rank} node_ranks:{node_ranks[node_rank]}"
    )

    # NOTE: there are different number of global mapped ranks on each node.
    free_ports = []
    trainer_endpoints = []
    for ip in node_ips:
        node_rank = node_ips.index(ip)
        if os.environ.get('PADDLE_PORT') is not None:
            start_port = int(os.getenv("PADDLE_PORT", ""))
            free_ports = list(
                range(start_port, start_port + len(node_ranks[node_rank]))
            )
        elif os.environ.get('FLAGS_START_PORT') is not None:
            start_port = int(os.environ.get('FLAGS_START_PORT'))
            free_ports = list(
                range(start_port, start_port + len(node_ranks[node_rank]))
            )
        else:
            free_ports = find_free_ports(len(node_ranks[node_rank]))
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])

    return get_mapped_cluster_with_rank_mapping(
        node_ips,
        node_ip,
        trainer_endpoints,
        device_mode,
        node_ranks,
        node_rank_mappings,
    )


class ParameterServerLauncher:
    def __init__(self, args, distribute_mode):
        self.args = args
        self.distribute_mode = distribute_mode
        self.with_coordinator = False
        self.server_num = 0
        self.worker_num = 0
        self.heter_worker_num = 0
        self.coordinator_num = 0

        self.server_endpoints = ""
        self.server_endpoints_ips = []
        self.server_endpoints_port = []

        self.worker_endpoints = ""
        self.worker_endpoints_ips = []
        self.worker_endpoints_port = []

        self.heter_worker_endpoints = ""
        self.heter_worker_endpoints_ips = []
        self.heter_worker_endpoints_port = []

        self.coordinator_endpoints = ""
        self.coordinator_endpoints_ips = []
        self.coordinator_endpoints_port = []

        self.is_local = True
        self.current_node_ip = ""

        self.stage_trainer_num = []
        self.stage_heter_map = {}
        self.stage_list = []
        self.stage_device_map = {}
        self.stage_num = 0

        self.get_role_endpoints(args)

    def get_role_endpoints(self, args):
        if args.server_num:
            self.server_num = args.server_num
            if args.servers:
                assert (
                    len(args.servers.split(",")) == self.server_num
                ), "The server_num and servers doesn't match. Expect servers endpoints num equal to server_num, but received servers endpoint num: {} and server_num {}".format(
                    len(args.servers.split(",")), self.server_num
                )
                self.server_endpoints = args.servers
            else:
                ports = get_ports(self.server_num, 0)
                self.server_endpoints = ",".join(
                    ["127.0.0.1:" + str(x) for x in ports]
                )
        else:
            assert (
                args.servers != ""
            ), "The setting of Parameter-Server must has server_num or servers."
            self.server_endpoints = args.servers
            self.server_num = len(self.server_endpoints.split(","))

        # get worker envs
        if args.worker_num:
            self.worker_num = args.worker_num
            if args.workers:
                assert (
                    len(args.workers.split(",")) == self.worker_num
                ), "The worker_num and workers doesn't match. Expect workers endpoints num equal to worker_num, but received workers endpoint num: {} and worker_num {}".format(
                    len(args.workers.split(",")), self.worker_num
                )

                self.worker_endpoints = args.workers
            else:
                ports = get_ports(self.worker_num, self.server_num)
                self.worker_endpoints = ",".join(
                    ["127.0.0.1:" + str(x) for x in ports]
                )
        else:
            assert (
                args.workers != ""
            ), "The setting of Parameter-Server must has worker_num or workers."
            worker_endpoints_ips = [
                x.strip().split(":")[0] for x in args.workers.split(",")
            ]
            self.worker_num = len(worker_endpoints_ips)
            worker_endpoints_len = [
                len(x.strip().split(":")) for x in args.workers.split(",")
            ]

            if 1 in worker_endpoints_len:
                # if no port value in worker_endpoints, will set default port values.
                start_port = 6170
                worker_endpoints_port = range(
                    start_port + self.server_num,
                    start_port + self.server_num + self.worker_num,
                    1,
                )
                # create endpoints str
                worker_endpoints = []
                for i in range(self.worker_num):
                    worker_endpoints.append(
                        ":".join(
                            (
                                worker_endpoints_ips[i],
                                str(worker_endpoints_port[i]),
                            )
                        )
                    )
                self.worker_endpoints = ",".join(worker_endpoints)
            else:
                self.worker_endpoints = args.workers

        # get coordinator envs
        if args.coordinator_num:
            self.with_coordinator = True
            self.coordinator_num = args.coordinator_num
            if args.coordinators:
                assert (
                    len(args.coordinators.split(",")) == self.coordinator_num
                ), "The coordinator_num and coordinators doesn't match. Expect coordinators endpoints num equal to coordinator_num, but received coordinator endpoint num: {} and coordinator_num {}".format(
                    len(args.coordinators.split(",")), self.coordinator_num
                )

                self.coordinator_endpoints = args.coordinators
            else:
                ports = get_ports(self.coordinator_num, 1)
                self.coordinator_endpoints = ",".join(
                    ["127.0.0.1:" + str(x) for x in ports]
                )
                print(">>> use default coordinator addr(only one process)")

        # get heter worker envs
        if self.distribute_mode == DistributeMode.PS_HETER:
            assert (
                args.heter_devices != ""
            ), "The setting of Parameter-Server heter mode must has heter_devices."
            self.stage_device_map[1] = "cpu"  # for cpu trainer
            heter_devices_list = args.heter_devices.split(";")
            for i in range(len(heter_devices_list)):
                self.stage_device_map[i + 2] = heter_devices_list[i]

            self.stage_heter_map[1] = self.worker_endpoints
            if args.heter_worker_num:
                self.stage_heter_trainer_num = args.heter_worker_num.split(";")
                self.stage_heter_trainer_num = [
                    int(trainer_num)
                    for trainer_num in self.stage_heter_trainer_num
                ]

                if args.heter_workers:
                    assert len(args.heter_workers.split(";")) == len(
                        self.stage_heter_trainer_num
                    ), "The stage_num and heter_workers doesn't match. Expect heter_workers endpoints stage num equal to heter_worker_num stage, but received heter_workers endpoint stage num: {} and heter_worker_num stage {}".format(
                        len(args.heter_workers.split(";")),
                        len(self.stage_heter_trainer_num),
                    )
                    heter_worker_endpoints_list = args.heter_workers.split(";")
                    self.heter_worker_endpoints = ""
                    for i in range(len(self.stage_heter_trainer_num)):
                        if self.heter_worker_endpoints != "":
                            self.heter_worker_endpoints += ","
                        heter_worker_endpoints = heter_worker_endpoints_list[
                            i
                        ].split(",")
                        assert (
                            len(heter_worker_endpoints)
                            == self.stage_heter_trainer_num[i]
                        ), f"The heter trainer num in stage {i} is not equal in args.heter_worker_num and args.heter_workers"

                        heter_worker_endpoints_ips = [
                            x.strip().split(":")[0]
                            for x in heter_worker_endpoints
                        ]
                        heter_worker_endpoints_len = [
                            len(x.strip().split(":"))
                            for x in heter_worker_endpoints
                        ]

                        if 1 in heter_worker_endpoints_len:
                            # if no port value in heter_worker_endpoint, will set default port values.
                            heter_worker_endpoints_port = get_ports(
                                len(heter_worker_endpoints_ips),
                                self.worker_num
                                + self.server_num
                                + self.heter_worker_num,
                            )
                            new_heter_worker_endpoints = []
                            for j in range(len(heter_worker_endpoints_ips)):
                                new_heter_worker_endpoints.append(
                                    ":".join(
                                        (
                                            heter_worker_endpoints_ips[j],
                                            str(heter_worker_endpoints_port[j]),
                                        )
                                    )
                                )
                            ip_port_list = ",".join(new_heter_worker_endpoints)
                        else:
                            ip_port_list = ",".join(heter_worker_endpoints)

                        self.stage_heter_map[i + 2] = ip_port_list
                        self.stage_list.extend(
                            [i + 2] * len(ip_port_list.split(','))
                        )

                        self.heter_worker_num += self.stage_heter_trainer_num[i]
                        self.heter_worker_endpoints += ip_port_list
                else:
                    for i in range(len(self.stage_heter_trainer_num)):
                        heter_trainer_num = self.stage_heter_trainer_num[i]
                        ports = get_ports(
                            heter_trainer_num,
                            self.server_num
                            + self.worker_num
                            + self.heter_worker_num,
                        )
                        ip_port_list = ",".join(
                            ["127.0.0.1:" + str(x) for x in ports]
                        )
                        self.stage_heter_map[i + 2] = ip_port_list
                        self.stage_list.extend(
                            [i + 2] * len(ip_port_list.split(','))
                        )
                        self.heter_worker_num += heter_trainer_num
                        if self.heter_worker_endpoints != "":
                            self.heter_worker_endpoints += ","
                        self.heter_worker_endpoints += ip_port_list
            else:
                assert (
                    args.heter_workers != ""
                ), "The setting of Parameter-Server heter mode must has heter_worker_num or heter_workers."
                self.stage_heter_trainer_num = []
                heter_worker_endpoints_list = args.heter_workers.split(";")
                self.heter_worker_endpoints = ""
                for i in range(len(heter_worker_endpoints_list)):
                    heter_worker_endpoints = heter_worker_endpoints_list[
                        i
                    ].split(",")
                    self.stage_heter_trainer_num.append(
                        len(heter_worker_endpoints)
                    )
                    heter_worker_endpoints_ips = [
                        x.strip().split(":")[0] for x in heter_worker_endpoints
                    ]
                    heter_worker_endpoints_len = [
                        len(x.strip().split(":"))
                        for x in heter_worker_endpoints
                    ]
                    if 1 in heter_worker_endpoints_len:
                        # if no port value in heter_worker_endpoint, will set default port values.
                        heter_worker_endpoints_port = get_ports(
                            len(heter_worker_endpoints_ips),
                            self.worker_num
                            + self.server_num
                            + self.heter_worker_num,
                        )

                        new_heter_worker_endpoints = []
                        for j in range(len(heter_worker_endpoints_ips)):
                            new_heter_worker_endpoints.append(
                                ":".join(
                                    (
                                        heter_worker_endpoints_ips[j],
                                        str(heter_worker_endpoints_port[j]),
                                    )
                                )
                            )
                        ip_port_list = ",".join(new_heter_worker_endpoints)
                    else:
                        ip_port_list = ",".join(heter_worker_endpoints)

                    self.stage_heter_map[i + 2] = ip_port_list
                    self.stage_list.extend(
                        [i + 2] * len(ip_port_list.split(','))
                    )

                    self.heter_worker_num += self.stage_heter_trainer_num[-1]
                    if self.heter_worker_endpoints != "":
                        self.heter_worker_endpoints += ","
                    self.heter_worker_endpoints += ip_port_list

            self.stage_trainer_num = [
                self.worker_num
            ] + self.stage_heter_trainer_num
            self.stage_num = len(self.stage_trainer_num)

        # get http_port
        if args.http_port:
            http_port = [args.http_port]
        else:
            http_port = get_ports(
                1, self.server_num + self.worker_num + self.heter_worker_num
            )
        http_ip = self.server_endpoints.split(",")[0].split(":")[0]
        self.http_port = http_ip + ":" + str(http_port[0])

        # check local or user define
        self.server_endpoints_ips = [
            x.strip().split(":")[0] for x in self.server_endpoints.split(",")
        ]
        self.worker_endpoints_ips = [
            x.strip().split(":")[0] for x in self.worker_endpoints.split(",")
        ]

        if self.with_coordinator:
            self.coordinator_endpoints_ips = [
                x.strip().split(":")[0]
                for x in self.coordinator_endpoints.split(",")
            ]
            self.coordinator_endpoints_port = [
                x.strip().split(":")[1]
                for x in self.coordinator_endpoints.split(",")
            ]

        self.server_endpoints_port = [
            x.strip().split(":")[1] for x in self.server_endpoints.split(",")
        ]
        self.worker_endpoints_port = [
            x.strip().split(":")[1] for x in self.worker_endpoints.split(",")
        ]
        self.node_ips = []
        for ip in self.server_endpoints_ips:
            if ip not in self.node_ips:
                self.node_ips.append(ip)
        for ip in self.worker_endpoints_ips:
            if ip not in self.node_ips:
                self.node_ips.append(ip)

        if self.distribute_mode == DistributeMode.PS_HETER:
            self.heter_worker_endpoints_ips = [
                x.strip().split(":")[0]
                for x in self.heter_worker_endpoints.split(",")
            ]
            self.heter_worker_endpoints_port = [
                x.strip().split(":")[1]
                for x in self.heter_worker_endpoints.split(",")
            ]
            for ip in self.heter_worker_endpoints_ips:
                if ip not in self.node_ips:
                    self.node_ips.append(ip)

        if len(set(self.node_ips)) == 1:
            self.is_local = True
            self.current_node_ip = self.node_ips[0]
        else:
            self.is_local = False
            pod_ip = os.getenv("POD_IP", None)
            if pod_ip is None:
                _, self.current_node_ip = get_host_name_ip()
            else:
                self.current_node_ip = pod_ip
            if not self.distribute_mode == DistributeMode.PS_HETER:
                assert (
                    self.current_node_ip in self.node_ips
                ), f"Can't find your local ip {{{self.current_node_ip}}} in args.servers and args.workers ips: {{{self.node_ips}}}"
        if self.current_node_ip in self.node_ips:
            self.node_rank = self.node_ips.index(self.current_node_ip)
            logger.debug(
                f"parsed from args: node_ips:{self.node_ips} current_node_ip:{self.current_node_ip} node_rank:{self.node_rank}"
            )

    def start_ps(self):
        if self.current_node_ip not in self.node_ips:
            return
        cluster = Cluster(hdfs=None)
        server_rank = 0
        worker_rank = 0
        heter_worker_rank = 0
        coordinator_rank = 0
        for node_rank, ip in enumerate(self.node_ips):
            pod = Pod()
            pod.rank = node_rank
            pod.addr = ip
            for i in range(len(self.server_endpoints_ips)):
                if ip == self.server_endpoints_ips[i]:
                    server = Trainer()
                    server.endpoint = f"{ip}:{self.server_endpoints_port[i]}"
                    server.rank = server_rank
                    server_rank += 1
                    pod.servers.append(server)
            for j in range(len(self.worker_endpoints_ips)):
                if ip == self.worker_endpoints_ips[j]:
                    worker = Trainer()
                    worker.endpoint = f"{ip}:{self.worker_endpoints_port[j]}"
                    worker.rank = worker_rank
                    worker.stage = 1
                    worker_rank += 1
                    pod.workers.append(worker)
            for m in range(len(self.coordinator_endpoints_ips)):
                if ip == self.coordinator_endpoints_ips[m]:
                    coordinator = Trainer()
                    coordinator.endpoint = (
                        f"{ip}:{self.coordinator_endpoints_port[m]}"
                    )
                    coordinator.rank = coordinator_rank
                    coordinator.stage = 1
                    coordinator_rank += 1
                    pod.coordinators.append(coordinator)

            for k in range(len(self.heter_worker_endpoints_ips)):
                if ip == self.heter_worker_endpoints_ips[k]:
                    heter_worker = Trainer()
                    heter_worker.endpoint = (
                        f"{ip}:{self.heter_worker_endpoints_port[k]}"
                    )
                    heter_worker.rank = heter_worker_rank
                    heter_worker.stage = self.stage_list[k]
                    heter_worker_rank += 1
                    pod.heter_workers.append(heter_worker)

            cluster.pods.append(pod)

        pod = cluster.pods[self.node_rank]
        self.gloo_rendezvous_dir = tempfile.mkdtemp()

        # 3. subprocess start
        self.procs = {
            "worker": [],
            "coordinator": [],
            "server": [],
            "heter_worker": [],
        }
        self.cmds = {
            "worker": [],
            "coordinator": [],
            "server": [],
            "heter_worker": [],
        }
        self.log_fns = {
            "worker": [],
            "coordinator": [],
            "server": [],
            "heter_worker": [],
        }

        self.start_pod_server(self.args, pod)
        self.start_pod_worker(self.args, pod)
        if self.with_coordinator:
            self.start_pod_coordinator(self.args, pod)
        if self.distribute_mode == DistributeMode.PS_HETER:
            self.start_pod_heter_worker(self.args, pod)

        logger.info(
            f"Please check servers, workers, coordinator and heter_worker logs in {self.args.log_dir}/workerlog.*, {self.args.log_dir}/serverlog.* , {self.args.log_dir}/coordinatorlog.*, and {self.args.log_dir}/heterlog.*"
        )

        # 4. wait for finish training
        if len(self.procs["worker"]) > 0:
            # if node has worker procs
            # only wait worker to finish here
            for i, proc in enumerate(self.procs["worker"]):
                self.procs["worker"][i].proc.wait()
                if len(self.log_fns["worker"]) > 0:
                    self.log_fns["worker"][i].close()
            logger.info(
                "all workers exit, going to finish parameter server and heter_worker."
            )
            if len(self.procs["heter_worker"]) > 0:
                for i, proc in enumerate(self.procs["heter_worker"]):
                    self.log_fns["heter_worker"][i].close()
                    self.procs["heter_worker"][i].proc.terminate()
                logger.info("all heter_worker are killed")

            if len(self.procs["server"]) > 0:
                for i, proc in enumerate(self.procs["server"]):
                    self.log_fns["server"][i].close()
                    self.procs["server"][i].proc.terminate()
                logger.info("all parameter server are killed")

            if len(self.procs["coordinator"]) > 0:
                for i, proc in enumerate(self.procs["coordinator"]):
                    self.log_fns["coordinator"][i].close()
                    self.procs["coordinator"][i].proc.terminate()
                logger.info("all coordinators are killed")

        else:
            # if node has not worker procs
            # blocking training process
            if len(self.procs["server"]) > 0:
                for i, proc in enumerate(self.procs["server"]):
                    self.procs["server"][i].proc.wait()

            if len(self.procs["heter_worker"]) > 0:
                for i, proc in enumerate(self.procs["heter_worker"]):
                    self.procs["heter_worker"][i].proc.wait()

        if os.path.exists(self.gloo_rendezvous_dir):
            shutil.rmtree(self.gloo_rendezvous_dir)

    def start_pod_server(self, args, pod):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)
        for idx, cur_server in enumerate(pod.servers):
            if self.distribute_mode == DistributeMode.PS_HETER:
                proc_env = {
                    "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                    "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                    "PADDLE_COORDINATOR_ENDPOINTS": self.coordinator_endpoints,
                    "PADDLE_ALL_HETER_TRAINER_IP_PORT_LIST": self.heter_worker_endpoints,
                    "PADDLE_PORT": cur_server.endpoint.split(":")[1],
                    "TRAINING_ROLE": "PSERVER",
                    "PADDLE_TRAINERS_NUM": str(self.worker_num),
                    "POD_IP": cur_server.endpoint.split(":")[0],
                    "PADDLE_WITH_GLOO": str(os.getenv("PADDLE_WITH_GLOO", "0")),
                    "PADDLE_GLOO_RENDEZVOUS": "3",
                    "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                    "PADDLE_GLOO_HTTP_ENDPOINT": self.http_port,
                }
            else:
                proc_env = {
                    "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                    "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                    "PADDLE_COORDINATOR_ENDPOINTS": self.coordinator_endpoints,
                    "PADDLE_PORT": cur_server.endpoint.split(":")[1],
                    "TRAINING_ROLE": "PSERVER",
                    "PADDLE_TRAINERS_NUM": str(self.worker_num),
                    "POD_IP": cur_server.endpoint.split(":")[0],
                    "PADDLE_WITH_GLOO": str(os.getenv("PADDLE_WITH_GLOO", "0")),
                    "PADDLE_GLOO_RENDEZVOUS": "3",
                    "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                    "PADDLE_GLOO_HTTP_ENDPOINT": self.http_port,
                }
            current_env.update(proc_env)

            cmd = [
                sys.executable,
                "-u",
                args.training_script,
            ] + args.training_script_args
            self.cmds["server"].append(cmd)

            if idx == 0:
                logger.info(
                    "Local server start {} processes. First process distributed "
                    "environment info (Only For Debug): {}".format(
                        len(pod.servers),
                        pretty_print_envs(
                            proc_env, ("Distributed Envs", "Value")
                        ),
                    )
                )

            if args.log_dir is not None:
                os.makedirs(args.log_dir, exist_ok=True)
                fn = open("%s/serverlog.%d" % (args.log_dir, idx), "w")
                self.log_fns["server"].append(fn)
                proc = subprocess.Popen(
                    cmd, env=current_env, stdout=fn, stderr=fn
                )
            else:
                proc = subprocess.Popen(cmd, env=current_env)

            tp = TrainerProc()
            tp.proc = proc
            tp.rank = cur_server.rank
            tp.local_rank = idx
            tp.log_fn = fn
            tp.log_offset = fn.tell() if fn else None
            tp.cmd = cmd

            self.procs["server"].append(tp)

    def start_pod_worker(self, args, pod):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        heter_device_num = 0
        device_list = []
        if framework.core.is_compiled_with_cuda():
            device_list = get_gpus(args.gpus)
            heter_device_num = len(device_list)
        elif framework.core.is_compiled_with_xpu():
            heter_device_num = framework.core.get_xpu_device_count()
            device_list = [str(x) for x in range(0, heter_device_num)]

        for idx, cur_worker in enumerate(pod.workers):
            device_id = (
                "0"
                if heter_device_num == 0
                else str(device_list[(idx) % heter_device_num])
            )
            if self.distribute_mode == DistributeMode.PS_HETER:
                proc_env = {
                    "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                    "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                    "PADDLE_TRAINERS_NUM": str(self.worker_num),
                    "PADDLE_COORDINATOR_ENDPOINTS": self.coordinator_endpoints,
                    "PADDLE_STAGE_TRAINERS_NUM": str(self.stage_trainer_num),
                    "STAGE_ID": "1",
                    "STAGE_NUM": str(self.stage_num),
                    "PADDLE_PREVIOUS_HETER_TRAINER_IP_PORT_LIST": "",
                    "PADDLE_NEXT_HETER_TRAINER_IP_PORT_LIST": self.stage_heter_map[
                        2
                    ],
                    "PADDLE_ALL_HETER_TRAINER_IP_PORT_LIST": self.heter_worker_endpoints,
                    "HETER_DEVICE_TYPE": self.stage_device_map[1],
                    "TRAINING_ROLE": "TRAINER",
                    "POD_IP": cur_worker.endpoint.split(":")[0],
                    "PADDLE_PORT": cur_worker.endpoint.split(":")[1],
                    "PADDLE_TRAINER_ID": str(cur_worker.rank),
                    "PADDLE_WITH_GLOO": str(os.getenv("PADDLE_WITH_GLOO", "0")),
                    "PADDLE_GLOO_RENDEZVOUS": "3",
                    "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                    "FLAGS_selected_gpus": "0",
                    "FLAGS_selected_xpus": "0",
                    "CUDA_VISIBLE_DEVICES": device_id,
                    "XPU_VISIBLE_DEVICES": device_id,
                    "PADDLE_GLOO_HTTP_ENDPOINT": self.http_port,
                }
            else:
                proc_env = {
                    "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                    "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                    "PADDLE_TRAINERS_NUM": str(self.worker_num),
                    "TRAINING_ROLE": "TRAINER",
                    "PADDLE_COORDINATOR_ENDPOINTS": self.coordinator_endpoints,
                    "POD_IP": cur_worker.endpoint.split(":")[0],
                    "PADDLE_PORT": cur_worker.endpoint.split(":")[1],
                    "PADDLE_TRAINER_ID": str(cur_worker.rank),
                    "PADDLE_WITH_GLOO": str(os.getenv("PADDLE_WITH_GLOO", "0")),
                    "PADDLE_GLOO_RENDEZVOUS": "3",
                    "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                    "FLAGS_selected_gpus": "0",
                    "FLAGS_selected_xpus": "0",
                    "CUDA_VISIBLE_DEVICES": device_id,
                    "XPU_VISIBLE_DEVICES": device_id,
                    "PADDLE_GLOO_HTTP_ENDPOINT": self.http_port,
                }

            current_env.update(proc_env)
            cmd = [
                sys.executable,
                "-u",
                args.training_script,
            ] + args.training_script_args
            self.cmds["worker"].append(cmd)

            if idx == 0:
                logger.info(
                    "Local worker start {} processes. First process distributed "
                    "environment info (Only For Debug): {}".format(
                        len(pod.workers),
                        pretty_print_envs(
                            proc_env, ("Distributed Envs", "Value")
                        ),
                    )
                )

            if args.log_dir is not None:
                os.makedirs(args.log_dir, exist_ok=True)
                fn = open(
                    "%s/workerlog.%d" % (args.log_dir, cur_worker.rank), "w"
                )
                self.log_fns["worker"].append(fn)
                proc = subprocess.Popen(
                    cmd, env=current_env, stdout=fn, stderr=fn
                )
            else:
                proc = subprocess.Popen(cmd, env=current_env)

            tp = TrainerProc()
            tp.proc = proc
            tp.rank = cur_worker.rank
            tp.local_rank = idx
            tp.log_fn = fn
            tp.log_offset = fn.tell() if fn else None
            tp.cmd = cmd

            self.procs["worker"].append(tp)

    def start_pod_coordinator(self, args, pod):
        print(">>> entering start_pod_coordinator")
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        for idx, cur_coordinator in enumerate(pod.coordinators):
            device_id = "0"
            proc_env = {
                "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                "PADDLE_TRAINERS_NUM": str(self.worker_num),
                "PADDLE_COORDINATOR_ENDPOINTS": self.coordinator_endpoints,
                "PADDLE_COORDINATOR_NUM": str(self.coordinator_num),
                "TRAINING_ROLE": "COORDINATOR",
                "POD_IP": cur_coordinator.endpoint.split(":")[0],
                "PADDLE_PORT": cur_coordinator.endpoint.split(":")[1],
                "PADDLE_TRAINER_ID": str(cur_coordinator.rank),
                "PADDLE_WITH_GLOO": str(os.getenv("PADDLE_WITH_GLOO", "0")),
                "PADDLE_GLOO_RENDEZVOUS": "3",
                "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                "FLAGS_selected_gpus": "0",
                "FLAGS_selected_xpus": "0",
                "CUDA_VISIBLE_DEVICES": device_id,
                "XPU_VISIBLE_DEVICES": device_id,
                "PADDLE_GLOO_HTTP_ENDPOINT": self.http_port,
            }

            current_env.update(proc_env)
            cmd = [
                sys.executable,
                "-u",
                args.training_script,
            ] + args.training_script_args
            self.cmds["coordinator"].append(cmd)

            if idx == 0:
                logger.info(
                    "Local coordinator start {} processes. First process distributed "
                    "environment info (Only For Debug): {}".format(
                        len(pod.coordinators),
                        pretty_print_envs(
                            proc_env, ("Distributed Envs", "Value")
                        ),
                    )
                )

            if args.log_dir is not None:
                os.makedirs(args.log_dir, exist_ok=True)
                fn = open("%s/coordinator.%d" % (args.log_dir, idx), "w")
                self.log_fns["coordinator"].append(fn)
                proc = subprocess.Popen(
                    cmd, env=current_env, stdout=fn, stderr=fn
                )
            else:
                proc = subprocess.Popen(cmd, env=current_env)

            tp = TrainerProc()
            tp.proc = proc
            tp.rank = cur_coordinator.rank
            tp.local_rank = idx
            tp.log_fn = fn
            tp.log_offset = fn.tell() if fn else None
            tp.cmd = cmd

            self.procs["coordinator"].append(tp)

    def start_pod_heter_worker(self, args, pod):
        default_env = os.environ.copy()
        current_env = copy.copy(default_env)
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        heter_device_num = 0
        device_list = []
        if framework.core.is_compiled_with_cuda():
            device_list = get_gpus(args.gpus)
            heter_device_num = len(device_list)
        elif framework.core.is_compiled_with_xpu():
            heter_device_num = framework.core.get_xpu_device_count()
            device_list = [str(x) for x in range(0, heter_device_num)]

        for idx, cur_heter_worker in enumerate(pod.heter_workers):
            device_id = (
                "0"
                if heter_device_num == 0
                else str(device_list[(idx) % heter_device_num])
            )
            stage_id = cur_heter_worker.stage
            proc_env = {
                "PADDLE_PSERVERS_IP_PORT_LIST": self.server_endpoints,
                "PADDLE_TRAINER_ENDPOINTS": self.worker_endpoints,
                "PADDLE_NEXT_HETER_TRAINER_IP_PORT_LIST": (
                    self.stage_heter_map[stage_id + 1]
                    if stage_id <= self.stage_num - 1
                    else ""
                ),
                "PADDLE_PREVIOUS_HETER_TRAINER_IP_PORT_LIST": self.stage_heter_map[
                    stage_id - 1
                ],
                "PADDLE_ALL_HETER_TRAINER_IP_PORT_LIST": self.heter_worker_endpoints,
                "HETER_DEVICE_TYPE": self.stage_device_map[stage_id],
                "STAGE_ID": str(stage_id),
                "STAGE_NUM": str(self.stage_num),
                "PADDLE_PORT": cur_heter_worker.endpoint.split(":")[1],
                "TRAINING_ROLE": "HETER_TRAINER",
                "PADDLE_TRAINERS_NUM": str(self.worker_num),
                "PADDLE_STAGE_TRAINERS_NUM": str(self.stage_trainer_num),
                "POD_IP": cur_heter_worker.endpoint.split(":")[0],
                "PADDLE_WITH_GLOO": str(os.getenv("PADDLE_WITH_GLOO", "0")),
                "PADDLE_GLOO_RENDEZVOUS": "3",
                "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                "FLAGS_selected_gpus": "0",
                "FLAGS_selected_xpus": "0",
                "CUDA_VISIBLE_DEVICES": device_id,
                "XPU_VISIBLE_DEVICES": device_id,
                "PADDLE_GLOO_HTTP_ENDPOINT": self.http_port,
            }
            current_env.update(proc_env)

            cmd = [
                sys.executable,
                "-u",
                args.training_script,
            ] + args.training_script_args
            self.cmds["heter_worker"].append(cmd)

            if idx == 0:
                logger.info(
                    "Local heter_worker start {} processes. First process distributed "
                    "environment info (Only For Debug): {}".format(
                        len(pod.heter_workers),
                        pretty_print_envs(
                            proc_env, ("Distributed Envs", "Value")
                        ),
                    )
                )

            if args.log_dir is not None:
                os.makedirs(args.log_dir, exist_ok=True)
                fn = open("%s/heterlog.%d" % (args.log_dir, idx), "w")
                self.log_fns["heter_worker"].append(fn)
                proc = subprocess.Popen(
                    cmd, env=current_env, stdout=fn, stderr=fn
                )
            else:
                proc = subprocess.Popen(cmd, env=current_env)

            tp = TrainerProc()
            tp.proc = proc
            tp.rank = cur_heter_worker.rank
            tp.local_rank = idx
            tp.log_fn = fn
            tp.log_offset = fn.tell() if fn else None
            tp.cmd = cmd

            self.procs["heter_worker"].append(tp)


def check_backend(backend):
    if backend not in [
        'nccl',
        'gloo',
        'bkcl',
        'auto',
        'heter',
        'xccl',
    ]:
        raise ValueError(
            "paddle.distributed initialize error, "
            "backend argument can only be one of "
            "'nccl', 'gloo', 'bkcl', 'auto', 'heter', 'xccl' "
            f"but got {backend}"
        )

    if backend == 'nccl' and not framework.core.is_compiled_with_cuda():
        raise ValueError(
            "paddle.distributed initialize error, "
            "your paddle is not compiled with cuda but you assign 'nccl' as backend."
        )

    if backend == 'bkcl' and not framework.core.is_compiled_with_xpu():
        raise ValueError(
            "paddle.distributed initialize error, "
            "your paddle is not compiled with xpu but you assign 'bkcl' as backend."
        )


def block_windows_and_macos(backend):
    if backend != 'gloo':
        return
    if utils.OS_NAME.startswith('darwin'):  # MACOS , block
        raise ValueError(
            "You are going to using gloo on macos, but currently is not supported"
        )
    if utils.IS_WINDOWS:  # MACOS , block
        raise ValueError(
            "You are going to using gloo on windows, but currently is not supported"
        )


def get_backend_by_compile_flag():
    if framework.core.is_compiled_with_cuda():
        return 'nccl'

    if framework.core.is_compiled_with_xpu():
        return 'bkcl'

    return 'gloo'
