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
    
    1. for single node four cards training
       python -m torch.distributed.launch --selected_gpus="0,1,2,3,4" \
         your_training_py (arg1 arg2 and all others)

    2. for mulitple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            python -m torch.distributed.launch --node_ips="192.168.0.16, 192.168.0.17" \
                --node_id=0 \
                your_training_py (arg1 arg2 and all others)

        on 192.168.0.17:
            python -m torch.distributed.launch --node_ips="192.168.0.16, 192.168.0.17" 
                --node_id=1 \
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
        "--node_ips",
        type=str,
        default="127.0.0.1",
        help="Paddle trainer ips, such as 192.168.0.16,192.168.0.17..")

    parser.add_argument(
        "--node_id",
        type=int,
        default=0,
        help="The trainer id of the node for multi-node distributed "
        "training")

    parser.add_argument(
        "--print_config",
        type=bool,
        default=True,
        help="Print the config or not")

    parser.add_argument(
        "--selected_gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="It's for gpu trainning and the trainning process will run on the selected_gpus,"
        "each process is bound to a single GPU.")

    parser.add_argument(
        "--split_log_path", type=str, help="The splited log path.")

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

    node_id = args.node_id
    node_ips = [x.strip() for x in args.node_ips.split(',')]
    assert node_id >= 0 and node_id < len(node_ips), \
            "node_id:{} must be in range of the node_ips:{}".format(node_id, node_ips)

    current_node_ip = node_ips[node_id]
    num_nodes = len(node_ips)
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
    procs = []
    cmds = []
    for i in range(0, selected_gpus_num):
        current_env.update({
            "FLAGS_selected_gpus": "%s" % selected_gpus[i],
            "PADDLE_TRAINER_ID": "%d" % (node_id * selected_gpus_num + i),
            "PADDLE_CURRENT_ENDPOINT": "%s:%d" % (current_node_ip, 6170 + i),
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": trainers_endpoints
        })

        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args

        cmds.append(cmd)

        if args.split_log_path is not None:
            fn = open("%s/workerlog.%d" % (args.split_log_path, i), "w")
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
