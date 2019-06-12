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
process on each trainning node for gpu trainning.

Usage:
    In both of single node training or multiple node training, this module 
launch a process on each of the given gpu card.

    1. for single node trainning with all visible gpu cards:
       python -m paddle.distributed.launch \
         your_training_py (arg1 arg2 and all others)
    
    2. for single node trainning with [0,4) cards
       python -m paddle.distributed.launch --selected_gpus="0,1,2,3" \
         your_training_py (arg1 arg2 and all others)

    3. for mulitple node training such as two node:192.168.0.16, 192.168.0.17
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
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle.fluid as fluid


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
see: http://www.paddlepaddle.org/documentation/docs/zh/1.2/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
And your train program must read environment variables below in order to let different
process init properly:
FLAGS_selected_gpus
PADDLE_TRAINER_ID
PADDLE_CURRENT_ENDPOINT
PADDLE_TRAINERS_NUM
PADDLE_TRAINER_ENDPOINTS
POD_IP (current node ip address, not needed for local training)
''')

    # Optional arguments for the launch helper
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
        "--started_port",
        type=int,
        default=6170,
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
        help="It's for gpu trainning and the trainning process will run on the selected_gpus,"
        "each process is bound to a single GPU. And if it's not setted, this module will use all the gpu cards for training."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        help="The path for each process's log.If it's not setted, the log will printed to default pipe."
    )

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def start_procs(args):
    """
    """
    procs = []
    log_fns = []

    default_env = os.environ.copy()

    current_node_ip = args.node_ip
    node_ips = [x.strip() for x in args.cluster_node_ips.split(',')]
    node_id = node_ips.index(current_node_ip)
    num_nodes = len(node_ips)

    if args.selected_gpus is None:
        gpus_num = fluid.core.get_cuda_device_count()
        selected_gpus = [str(x) for x in range(0, gpus_num)]
    else:
        selected_gpus = [x.strip() for x in args.selected_gpus.split(',')]
    selected_gpus_num = len(selected_gpus)

    trainers_endpoints = ""
    for ip in node_ips:
        for i in range(selected_gpus_num):
            if trainers_endpoints != "":
                trainers_endpoints += ","
            trainers_endpoints += "%s:617%d" % (ip, i)

    nranks = num_nodes * selected_gpus_num

    if args.print_config:
        print("trainers_endpoints:", trainers_endpoints, ", node_id:", node_id,
              ", current_node_ip:", current_node_ip, ", num_nodes:", num_nodes,
              ", node_ips:", node_ips, ", nranks:", nranks)

    current_env = copy.copy(default_env)
    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    cmds = []
    for i in range(0, selected_gpus_num):
        current_env.update({
            "FLAGS_selected_gpus": "%s" % selected_gpus[i],
            "PADDLE_TRAINER_ID": "%d" % (node_id * selected_gpus_num + i),
            "PADDLE_CURRENT_ENDPOINT":
            "%s:%d" % (current_node_ip, args.started_port + i),
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": trainers_endpoints
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

    for i in range(0, len(procs)):
        proc = procs[i]

        proc.wait()
        if len(log_fns) > 0:
            log_fns[i].close()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=procs[i].returncode, cmd=cmds[i])


def launch():
    args = _parse_args()
    if args.print_config:
        _print_arguments(args)
    start_procs(args)


if __name__ == "__main__":
    launch()
