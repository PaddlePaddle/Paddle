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
"""
fleetrun is a module that spawns multiple distributed
process on each training node for gpu training and cpu training.
Usage:
    In both of single node training or multiple node training, this module
launch a process on each of the given gpu card or cpu machine.
    GPU training:
    1. for single node training with all visible gpu cards:
       fleetrun your_training_py (arg1 arg2 and all others)
    2. for single node training with [0,4) cards
       fleetrun --gpus="0,1,2,3" your_training_py (arg1 arg2 and all others)
    3. for multiple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    CPU training:
    1. for single node training with multi servers and workers:
        fleetrun --server_num=2 --worker_num=2 your_training_py (arg1 arg2 and all others)
    2. for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers.
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6171" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    3. use gloo backend for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers. (workers should set port)
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
"""

from __future__ import print_function
import sys
from sys import version
import subprocess
import os
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle
import paddle.fluid as fluid

from paddle.distributed.fleet.launch_utils import *
import paddle.distributed.fleet.cloud_utils as cloud_utils


def _print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def _parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description='''start paddle training using multi-process mode.
see: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
''')

    #Optional arguments for the launch helper
    parser.add_argument(
        "--ips",
        type=str,
        default="127.0.0.1",
        help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="It's for gpu training and the training process will run on the gpus,"
        "each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training."
    )

    parser.add_argument(
        "--servers", type=str, default="", help="User defined servers ip:port")
    parser.add_argument(
        "--workers", type=str, default="", help="User defined workers ip:port")
    parser.add_argument("--worker_num", type=int, help="number of workers")

    parser.add_argument("--server_num", type=int, help="number of servers")

    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="The path for each process's log.If it's not set, the log will printed to default pipe."
    )
    #positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script")

    #rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def get_cluster_from_args(args, gpus):
    node_ips = [x.strip() for x in args.ips.split(',')]
    if len(node_ips) == 1:
        node_ip = node_ips[0]
    else:
        _, node_ip = get_host_name_ip()

    # node_ip = args.node_ip
    assert node_ip in node_ips, "Can't find your local ip {%s} in node_ips: {%s}" \
                % (node_ip, node_ips)
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args: node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    free_ports = None
    if not cloud_utils.use_paddlecloud() and len(
            node_ips) <= 1 and os.environ.get('FLAGS_START_PORT') is None:
        free_ports = find_free_ports(len(gpus))
        if free_ports is not None:
            free_ports = list(free_ports)
    else:
        start_port = 6070
        if os.environ.get('FLAGS_START_PORT') is not None:
            start_port = os.environ.get('FLAGS_START_PORT')

        free_ports = [x for x in range(start_port, start_port + len(gpus))]

    return get_cluster(node_ips, node_ip, free_ports, gpus)


def get_gpus(gpus):
    if gpus is None:
        gpus_num = fluid.core.get_cuda_device_count()
        gpus = [str(x) for x in range(0, gpus_num)]
    else:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or cuda_visible_devices == "":
            gpus = [x.strip() for x in gpus.split(',')]
        else:
            # change gpus into relative values
            # e.g. CUDA_VISIBLE_DEVICES=4,5,6,7; args.gpus=4,5,6,7;
            # therefore gpus=0,1,2,3
            cuda_visible_devices_list = cuda_visible_devices.split(',')
            for x in gpus.split(','):
                assert x in cuda_visible_devices_list, "Can't find "\
                "your gpus %s in CUDA_VISIBLE_DEVICES[%s]."\
                % (x, cuda_visible_devices)
            gpus = [
                cuda_visible_devices_list.index(x.strip())
                for x in gpus.split(',')
            ]

    return gpus


def launch_collective(args):
    # parse arguments, used for cloud-single-machine and local
    gpus = get_gpus(args.gpus)
    trainers_num = cloud_utils.get_trainers_num()
    logger.debug("parsed from args trainerss_num:{} gpus:{}".format(
        trainers_num, gpus))

    cluster = None
    pod = None

    start_port = 6170
    if os.environ.get('FLAGS_START_PORT') is not None:
        start_port = os.environ.get('FLAGS_START_PORT')
    if cloud_utils.use_paddlecloud() and trainers_num != 1:
        cluster, pod = cloud_utils.get_cloud_cluster(args.ips, gpus, start_port)
        logger.info("get cluster from cloud:{}".format(cluster))
    else:
        # trainers_num = 1 or not use paddlecloud ips="a,b"
        cluster, pod = get_cluster_from_args(args, gpus)
        logger.info("get cluster from args:{}".format(cluster))

    procs = start_local_trainers(
        cluster,
        pod,
        training_script=args.training_script,
        training_script_args=args.training_script_args,
        log_dir=args.log_dir)

    while True:
        alive = watch_local_trainers(procs, cluster.trainers_nranks())

        if not alive:
            logger.info("Local procs complete, POD info:{}".format(pod))
            break

        time.sleep(3)


def launch_ps(args):
    ports = None
    start_port = 6170
    if args.server_num:
        server_num = args.server_num
        ports = get_ports(server_num, 0)
        server_endpoints = ",".join(["127.0.0.1:" + str(x) for x in ports])
    else:
        assert args.servers != "", "The setting of CPU mode must be either server_num or servers."
        server_endpoints = args.servers
    server_endpoints_ips = [
        x.strip().split(":")[0] for x in server_endpoints.split(",")
    ]
    server_endpoints_port = [
        x.strip().split(":")[1] for x in server_endpoints.split(",")
    ]
    server_num = len(server_endpoints_ips)

    if args.worker_num:
        worker_num = args.worker_num
        ports = get_ports(worker_num, server_num)
        worker_endpoints = ",".join(["127.0.0.1:" + str(x) for x in ports])
    else:
        assert args.workers != "", "The setting of CPU mode must be either worker_num or workers."
        worker_endpoints = args.workers
    worker_endpoints_ips = [
        x.strip().split(":")[0] for x in worker_endpoints.split(",")
    ]
    worker_num = len(worker_endpoints_ips)
    node_ips = list(set(server_endpoints_ips + worker_endpoints_ips))
    worker_endpoints_len = [
        len(x.strip().split(":")) for x in worker_endpoints.split(",")
    ]
    if 1 in worker_endpoints_len:
        # if no port value in worker_endpoints, will set default port values.
        worker_endpoints_port = range(start_port + server_num,
                                      start_port + server_num + worker_num, 1)
    else:
        worker_endpoints_port = [
            x.strip().split(":")[1] for x in worker_endpoints.split(",")
        ]

    # local train
    if len(set(node_ips)) == 1:
        current_node_ip = node_ips[0]
    else:
        _, current_node_ip = get_host_name_ip()

    assert current_node_ip in node_ips, "Can't find your local ip {%s} in args.servers and args.workers ips: {%s}" \
                % (current_node_ip, node_ips)
    node_rank = node_ips.index(current_node_ip)
    logger.debug(
        "parsed from args: node_ips:{} current_node_ip:{} node_rank:{}, server_ports:{}".
        format(node_ips, current_node_ip, node_rank, server_endpoints_port))

    cluster = Cluster(hdfs=None)
    server_rank = 0
    worker_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        for i in range(len(server_endpoints_ips)):
            if ip == server_endpoints_ips[i]:
                server = Trainer()
                server.endpoint = "%s:%s" % (ip, server_endpoints_port[i])
                server.rank = server_rank
                server_rank += 1
                pod.servers.append(server)
        for j in range(len(worker_endpoints_ips)):
            if ip == worker_endpoints_ips[j]:
                worker = Trainer()
                worker.endpoint = "%s:%s" % (ip, worker_endpoints_port[i])
                worker.rank = worker_rank
                worker_rank += 1
                pod.workers.append(worker)

        cluster.pods.append(pod)

    pod_rank = node_ips.index(current_node_ip)
    pod = cluster.pods[pod_rank]

    default_env = os.environ.copy()
    current_env = copy.copy(default_env)
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)
    procs = []
    cmds = []
    log_fns = []
    for idx, cur_server in enumerate(pod.servers):
        current_env.update({
            "PADDLE_PSERVERS_IP_PORT_LIST": server_endpoints,
            "PADDLE_PORT": cur_server.endpoint.split(":")[1],
            "TRAINING_ROLE": "PSERVER",
            "PADDLE_TRAINERS_NUM": str(worker_num),
            "POD_IP": cur_server.endpoint.split(":")[0]
        })

        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args
        cmds.append(cmd)

        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/serverlog.%d" % (args.log_dir, idx), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = cur_server.rank
        tp.local_rank = idx
        tp.log_fn = fn
        tp.log_offset = 0 if fn else None
        tp.cmd = cmd

        procs.append(tp)

    for idx, cur_worker in enumerate(pod.workers):
        current_env.update({
            "PADDLE_PSERVERS_IP_PORT_LIST": server_endpoints,
            "PADDLE_TRAINERS_NUM": str(worker_num),
            "TRAINING_ROLE": "TRAINER",
            "PADDLE_TRAINER_ID": str(cur_worker.rank)
        })
        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args
        cmds.append(cmd)
        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/workerlog.%d" % (args.log_dir, idx), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = cur_worker.rank
        tp.local_rank = idx
        tp.log_fn = fn
        tp.log_offset = 0 if fn else None
        tp.cmd = cmd

        procs.append(tp)

    # only wait worker to finish here
    for i, proc in enumerate(procs):
        if i < len(pod.servers):
            continue
        procs[i].proc.wait()
        if len(log_fns) > 0:
            log_fns[i].close()

    print("all workers exit, going to finish parameter server", file=sys.stderr)
    for i in range(len(pod.servers)):
        if len(log_fns) > 0:
            log_fns[i].close()
        procs[i].proc.terminate()
    print("all parameter server are killed", file=sys.stderr)


def launch():
    args = _parse_args()
    logger = get_logger()
    _print_arguments(args)
    ps_args = ['--worker_num', '--server_num', '--servers', '--workers']
    collective_args = ['--ips', '--gpus']
    has_ps_args = [
        ps_arg for ps_arg in ps_args if ps_arg in " ".join(sys.argv[1:-1])
    ]
    has_collective_args = [
        co_arg for co_arg in collective_args
        if co_arg in " ".join(sys.argv[1:-1])
    ]
    cuda_device_num = fluid.core.get_cuda_device_count()
    if len(has_ps_args) > 0 or cuda_device_num == 0:
        logger.info(
            "Run parameter-sever cpu mode. pserver args:{}, cuda count:{}".
            format(has_ps_args, cuda_device_num))
        launch_ps(args)
    elif len(has_collective_args) > 0:
        logger.info("Run collective gpu mode. gpu args:{}, cuda count:{}".
                    format(has_collective_args, cuda_device_num))
        launch_collective(args)
    else:
        logger.warning(
            "Not found distinct args. Default use gpu collective mode")
        launch_collective(args)


if __name__ == "__main__":
    launch()
