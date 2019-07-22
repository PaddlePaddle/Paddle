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
from __future__ import unicode_literals
import subprocess
import sys
import os
import copy
from argparse import ArgumentParser, REMAINDER


def parse_args():
    # Optional arguments for the launch helper
    parser = ArgumentParser(description="Distributed training")
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
        "--start_port",
        type=int,
        default=6170,
        help="The trainer's start port on a single node")

    parser.add_argument(
        "--print_config",
        type=bool,
        default=True,
        help="Print the config or not")

    parser.add_argument(
        "--worker_num", type=int, default=2, help="number of workers")

    parser.add_argument(
        "--server_num", type=int, default=2, help="number of servers")

    parser.add_argument(
        "--log_dir",
        default="logs",
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
    worker_num = args.worker_num
    server_num = args.server_num
    start_port = args.start_port
    default_env = os.environ.copy()
    current_env = copy.copy(default_env)
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)
    procs = []
    cmds = []
    log_fns = []
    ports = range(start_port, start_port + server_num, 1)
    endpoints = ",".join(["127.0.0.1:" + str(x) for x in ports])
    for i in range(server_num):
        current_env.update({
            "TRAINER_NUM": str(worker_num),
            "CURRENT_ID": str(i),
            "ENDPOINTS": endpoints,
            "TRAINING_ROLE": "PSERVER"
        })
        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args
        cmds.append(cmd)
        print(cmd)
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
            "ENDPOINTS": endpoints,
            "TRAINER_NUM": str(worker_num),
            "TRAINING_ROLE": "TRAINER",
            "CURRENT_ID": str(i)
        })
        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args
        print(cmd)
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
    args = parse_args()
    if args.print_config:
        start_procs(args)


# server num, worker num        
if __name__ == "__main__":
    launch()
