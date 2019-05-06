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

from __future__ import print_function
import sys
import subprocess
import os
import six
import copy
from argparse import ArgumentParser, REMAINDER


def _print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
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
        help="Paddle trainer ips, such as 192.168.0.16,192.168.0.17..")

    parser.add_argument(
        "--node_id",
        type=int,
        help="The trainer id of the node for multi-node distributed "
        "training")

    parser.add_argument(
        "--print_config",
        type=bool,
        default=True,
        help="Print the config or not")

    parser.add_argument(
        "--current_node_ip", type=str, help="The ip of current node.")

    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help="The number of process to use on each node "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.")

    parser.add_argument(
        "--selected_gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="The number of process to use on each node "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.")

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
    procs = []
    log_fns = []

    default_env = os.environ.copy()

    node_id = args.node_id
    node_ips = [x.strip() for x in args.node_ips.split(',')]
    current_ip = args.current_node_ip
    num_nodes = len(node_ips)
    selected_gpus = [x.strip() for x in args.selected_gpus.split(',')]
    selected_gpu_num = len(selected_gpus)

    all_trainer_endpoints = ""
    for ip in node_ips:
        for i in range(args.nproc_per_node):
            if all_trainer_endpoints != "":
                all_trainer_endpoints += ","
            all_trainer_endpoints += "%s:617%d" % (ip, i)

    nranks = num_nodes * args.nproc_per_node
    gpus_per_proc = args.nproc_per_node % selected_gpu_num
    if gpus_per_proc == 0:
        gpus_per_proc = selected_gpu_num / args.nproc_per_node
    else:
        gpus_per_proc = selected_gpu_num / args.nproc_per_node + 1

    selected_gpus_per_proc = [
        selected_gpus[i:i + gpus_per_proc]
        for i in range(0, len(selected_gpus), gpus_per_proc)
    ]

    if args.print_config:
        print("all_trainer_endpoints:", all_trainer_endpoints, ", node_id:",
              node_id, ", current_ip:", current_ip, ", num_nodes:", num_nodes,
              ", node_ips:", node_ips, ", gpus_per_proc:", gpus_per_proc,
              ", selected_gpus_per_proc:", selected_gpus_per_proc, ", nranks:",
              nranks)

    current_env = copy.copy(default_env)
    processes = []
    for i in range(0, args.nproc_per_node):
        current_env.update({
            "FLAGS_selected_gpus":
            "%s" % ",".join([str(s) for s in selected_gpus_per_proc[i]]),
            "PADDLE_TRAINER_ID": "%d" % (node_id * args.nproc_per_node + i),
            "PADDLE_CURRENT_ENDPOINT": "%s:617%d" % (current_ip, i),
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": all_trainer_endpoints
        })

        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=process.args)


def launch():
    args = _parse_args()
    if args.print_config:
        _print_arguments(args)
    start_procs(args)


if __name__ == "__main__":
    launch()
