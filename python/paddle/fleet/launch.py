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
paddle.distributed.launch is a module that spawns multiple distributed 
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
            fleetrun --ips="192.168.0.16,192.168.0.17" --node_ip=192.168.0.16 \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                --node_ip=192.168.0.17 \
                your_training_py (arg1 arg2 and all others)
    CPU training:
    1. for single node training with multi servers and workers:
        fleetrun --server_num=1 --worker_num=4 your_training_py (arg1 arg2 and all others)
    2. for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and  4 workers.
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6171" \
                --workers="192.168.0.16:6172,192.168.0.17:6173,192.168.0.16:6174,192.168.0.17:6175" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6171" \
                --workers="192.168.0.16:6172,192.168.0.17:6173,192.168.0.16:6174,192.168.0.17:6175" \
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

from paddle.fleet.launch_utils import *
import paddle.fleet.cloud_utils as cloud_utils


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
    parser.add_argument(
        "--worker_num", type=int, default=2, help="number of workers")

    parser.add_argument(
        "--server_num", type=int, default=2, help="number of servers")

    parser.add_argument(
        "--log_dir",
        type=str,
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
    assert node_ip in node_ips, "Can't find your local ip {%s} in node_ips:{%s}" \
                % (node_ip, node_ips)
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args:node_ips:{} node_ip:{} node_rank:{}".format(
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

    if cloud_utils.use_paddlecloud() and trainers_num != 1:
        cluster, pod = cloud_utils.get_cloud_cluster(args.ips, gpus)
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
    worker_num = args.worker_num
    server_num = args.server_num
    start_port = 6170
    if os.environ.get('FLAGS_START_PORT') is not None:
        start_port = os.environ.get('FLAGS_START_PORT')
    default_env = os.environ.copy()
    current_env = copy.copy(default_env)
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)
    procs = []
    cmds = []
    log_fns = []
    ports = range(start_port, start_port + server_num, 1)
    default_endpoints = ",".join(["127.0.0.1:" + str(x) for x in ports])
    user_endpoints = ""
    if args.servers == "":
        user_endpoints = default_endpoints
    else:
        user_endpoints = args.servers
    user_endpoints_ips = [x.split(":")[0] for x in user_endpoints.split(",")]
    user_endpoints_port = [x.split(":")[1] for x in user_endpoints.split(",")]
    for i in range(server_num):
        current_env.update({
            "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
            "PADDLE_PORT": user_endpoints_port[i],
            "TRAINING_ROLE": "PSERVER",
            "PADDLE_TRAINERS_NUM": str(worker_num),
            "POD_IP": user_endpoints_ips[i]
        })

        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args
        cmds.append(cmd)
        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/serverlog.%d" % (args.log_dir, i), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)
        procs.append(proc)

    for i in range(worker_num):
        current_env.update({
            "PADDLE_PSERVERS_IP_PORT_LIST": user_endpoints,
            "PADDLE_TRAINERS_NUM": str(worker_num),
            "TRAINING_ROLE": "TRAINER",
            "PADDLE_TRAINER_ID": str(i)
        })
        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args
        cmds.append(cmd)
        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/workerlog.%d" % (args.log_dir, i), "w")
            log_fns.append(fn)
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)
        procs.append(proc)

    # only wait worker to finish here
    for i, proc in enumerate(procs):
        if i < server_num:
            continue
        procs[i].wait()
        if len(log_fns) > 0:
            log_fns[i].close()

    print("all workers exit, going to finish parameter server", file=sys.stderr)
    for i in range(server_num):
        if len(log_fns) > 0:
            log_fns[i].close()
        procs[i].terminate()
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
    if len(has_ps_args) > 0 or fluid.core.get_cuda_device_count() == 0:
        logger.info("Run cpu parameter-sever mode.")
        launch_ps(args)
    elif len(has_collective_args) > 0:
        logger.info("Run gpu collective mode.")
        launch_collective(args)
    else:
        logger.warning(
            "Not found distinct args. Default use gpu collective mode")
        launch_collective(args)


if __name__ == "__main__":
    launch()
