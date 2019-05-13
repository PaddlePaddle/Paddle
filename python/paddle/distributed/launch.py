# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import subprocess
import os
import sys
import time
import argparse

default_envs = {
    "PADDLE_TRAINER_ENDPOINTS":
    "127.0.0.1:6170,127.0.0.1:6171,127.0.0.1:6172,127.0.0.1:6173,127.0.0.1:6174,127.0.0.1:6175,127.0.0.1:6176,127.0.0.1:6177",
    "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
    "PATH": os.getenv("PATH"),
    "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
    "PADDLE_TRAINERS_NUM": "8",
    "NCCL_DEBUG": "INFO",
    "GLOG_v": "0",
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_RETRY_CNT": "0",
    "PYTHONPATH": os.getenv("PYTHONPATH", ""),
}

GPUS = 8


def start_procs(gpus, entrypoint, entrypoint_args, log_dir):
    procs = []
    log_fns = []
    os.system("mkdir -p %s" % log_dir)
    # ======== update parent envs =======
    for k, v in os.environ.items():
        if k.startswith("FLAGS_") or k.startswith("NCCL_") or \
            k.startswith("GLOG_"):
            default_envs[k] = v

    # ======== for dist training =======
    node_trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    current_ip = os.getenv("POD_IP", "127.0.0.1")
    trainer_ips = os.getenv("PADDLE_TRAINERS", current_ip).split(",")
    num_nodes = len(trainer_ips)
    all_nodes_devices_endpoints = ""
    for n in trainer_ips:
        for i in range(gpus):
            if all_nodes_devices_endpoints:
                all_nodes_devices_endpoints += ","
            all_nodes_devices_endpoints += "%s:617%d" % (n, i)
    nranks = num_nodes * gpus
    # ======== for dist training =======

    for i in range(gpus):
        curr_env = {}
        curr_env.update(default_envs)
        curr_env.update({
            "FLAGS_selected_gpus": "%d" % i,
            "PADDLE_TRAINER_ID": "%d" % (node_trainer_id * gpus + i),
            "PADDLE_CURRENT_ENDPOINT": "%s:617%d" % (current_ip, i),
            # nranks
            "PADDLE_TRAINERS_NUM": "%d" % nranks,
            "PADDLE_TRAINER_ENDPOINTS": all_nodes_devices_endpoints
        })

        print("starting process ", i, entrypoint, entrypoint_args, curr_env)
        fn = open("%s/workerlog.%d" % (log_dir, i), "w")
        log_fns.append(fn)
        cmd = [sys.executable, "-u", entrypoint] + entrypoint_args
        procs.append(subprocess.Popen(cmd, stdout=fn, stderr=fn, env=curr_env))

    for i in range(gpus):
        try:
            procs[i].communicate()
            procs[i].terminate()
            log_fns[i].close()
        except:
            pass


def parse_args():

    parser = argparse.ArgumentParser(
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
    parser.add_argument(
        '--gpus',
        type=int,
        default=8,
        help='start number of processes for every gpu')
    parser.add_argument(
        '--log_dir',
        type=str,
        default="mylog",
        help='directory to put logs per process.')
    parser.add_argument(
        'entrypoint_script',
        type=str,
        help="The entrypoint script to be launched in parallel,"
        "followed by all the arguments for each process,"
        "e.g. train.py --lr 0.1")
    parser.add_argument('entrypoint_args', nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # launch multiple training process
    start_procs(args.gpus, args.entrypoint_script, args.entrypoint_args,
                args.log_dir)


if __name__ == "__main__":
    main()
