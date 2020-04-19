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
process on each training node for gpu training.
Usage:
    In both of single node training or multiple node training, this module 
launch a process on each of the given gpu card.
    1. for single node training with all visible gpu cards:
       python -m paddle.distributed.launch \
         your_training_py (arg1 arg2 and all others)
    
    2. for single node training with [0,4) cards
       python -m paddle.distributed.launch --selected_gpus="0,1,2,3" \
         your_training_py (arg1 arg2 and all others)
    3. for multiple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            python -m paddle.distributed.launch --cluster_node_ips="192.168.0.16,192.168.0.17" \
                --node_ip=192.168.0.16 \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            python -m paddle.distributed.launch --cluster_node_ips="192.168.0.16,192.168.0.17" \
                --node_ip=192.168.0.17 \
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

from paddle.distributed.utils import *
import paddle.distributed.cloud_utils as cloud_utils


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
NOTE: your train program ***must*** run as distributed nccl2 mode,
see: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
And your train program must read environment variables below in order to let different
process init properly:
FLAGS_selected_gpus
PADDLE_TRAINER_ID
PADDLE_CURRENT_ENDPOINT
PADDLE_TRAINERS_NUM
PADDLE_TRAINER_ENDPOINTS
POD_IP (current node ip address, not needed for local training)
''')

    #Optional arguments for the launch helper
    parser.add_argument(
        "--cluster_node_ips",
        type=str,
        default="127.0.0.1",
        help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")
    parser.add_argument(
        "--node_ip",
        type=str,
        default="127.0.0.1",
        help="The current node ip. ")
    parser.add_argument(
        "--use_paddlecloud",
        action='store_true',
        help="wheter to use paddlecloud platform to run your multi-process job. If false, no need to set this argument."
    )
    parser.add_argument(
        "--started_port",
        type=int,
        default=None,
        help="The trainer's started port on a single node")

    parser.add_argument(
        "--print_config",
        type=bool,
        default=True,
        help="Print the config or not")

    parser.add_argument(
        "--selected_gpus",
        type=str,
        default=None,
        help="It's for gpu training and the training process will run on the selected_gpus,"
        "each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training."
    )

    parser.add_argument(
        "--log_level",
        type=int,
        default=20,  # logging.INFO, details are here:https://docs.python.org/3/library/logging.html#levels
        help="Logging level, default is logging.INFO")

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


def get_cluster_from_args(args, selected_gpus):
    node_ips = [x.strip() for x in args.cluster_node_ips.split(',')]
    node_ip = args.node_ip
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args:node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    free_ports = None
    if not args.use_paddlecloud and len(
            node_ips) <= 1 and args.started_port is None:
        free_ports = find_free_ports(len(selected_gpus))
        if free_ports is not None:
            free_ports = list(free_ports)
    else:
        started_port = 6070
        if args.started_port is not None:
            started_port = args.started_port

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
            # change selected_gpus into relative values
            # e.g. CUDA_VISIBLE_DEVICES=4,5,6,7; args.selected_gpus=4,5,6,7;
            # therefore selected_gpus=0,1,2,3
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


def launch(args):
    # parse arguments, used for cloud-single-machine and local
    selected_gpus = get_gpus(args.selected_gpus)
    trainers_num = cloud_utils.get_trainers_num()
    logger.debug("parsed from args trainerss_num:{} selected_gpus:{}".format(
        trainers_num, selected_gpus))

    cluster = None
    pod = None

    if args.use_paddlecloud and trainers_num != 1:
        cluster, pod = cloud_utils.get_cloud_cluster(
            args.cluster_node_ips, args.node_ip, args.started_port,
            selected_gpus)
        logger.info("get cluster from cloud:{}".format(cluster))
    else:
        cluster, pod = get_cluster_from_args(args, selected_gpus)
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


if __name__ == "__main__":
    args = _parse_args()

    logger = get_logger(args.log_level)

    if args.print_config:
        _print_arguments(args)

    launch(args)
