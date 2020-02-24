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
import logging
import sys
from sys import version
import subprocess
import os
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle.fluid as fluid

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_format = logging.Formatter(
    '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
log_handler.setFormatter(log_format)
logger.addHandler(log_handler)


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
        help="It's for gpu training and the training process will run on the selected_gpus,"
        "each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training."
    )

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


def terminate_procs(procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()


def start_procs(args):
    """
    """
    default_env = os.environ.copy()

    current_node_ip = args.node_ip
    node_ips = [x.strip() for x in args.cluster_node_ips.split(',')]
    node_id = node_ips.index(current_node_ip)
    if args.use_paddlecloud:
        trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        if trainer_nums != 1:
            #you can automatically get ip info while using paddlecloud multi nodes mode.
            current_node_ip = os.getenv("POD_IP")
            assert current_node_ip is not None, "POD_IP should not be None"
            node_ips = os.getenv("PADDLE_TRAINERS")
            assert node_ips is not None, "PADDLE_TRAINERS should not be None"
            node_ips = node_ips.split(",")
            node_id = os.getenv("PADDLE_TRAINER_ID")
            assert node_id is not None, "PADDLE_TRAINER_ID should not be None"
            node_id = int(node_id)

            if args.node_ip != "127.0.0.1" and current_node_ip != args.node_ip:
                logger.warning(
                    "Please NOTE: When using paddlecloud, current_node_ip is \
automatically got from POD_IP. Your input node_ip: {} doesn't equals to \
current_node_ip: {} from paddlecloud environment."
                    .format(args.node_ip, current_node_ip))
            if args.cluster_node_ips != "127.0.0.1" and args.cluster_node_ips != ",".join(
                    node_ips):
                logger.warning(
                    "Please NOTE: When using paddlecloud, cluster_node_ips is \
automatically got from PADDLE_TRAINERS(multi nodes) or POD_IP(single node).\
Your input cluster_node_ips: {} doesn't equals to IPs: {} from \
paddlecloud environment.".format(args.cluster_node_ips, node_ips))
    num_nodes = len(node_ips)

    if args.selected_gpus is None:
        gpus_num = fluid.core.get_cuda_device_count()
        selected_gpus = [str(x) for x in range(0, gpus_num)]
    else:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None or cuda_visible_devices == "":
            selected_gpus = [x.strip() for x in args.selected_gpus.split(',')]
        else:
            # change selected_gpus into relative values
            # e.g. CUDA_VISIBLE_DEVICES=4,5,6,7; args.selected_gpus=4,5,6,7;
            # therefore selected_gpus=0,1,2,3
            cuda_visible_devices_list = cuda_visible_devices.split(',')
            for x in args.selected_gpus.split(','):
                assert x in cuda_visible_devices_list, "Can't find "\
                "your selected_gpus %s in CUDA_VISIBLE_DEVICES[%s]."\
                % (x, cuda_visible_devices)
            selected_gpus = [
                cuda_visible_devices_list.index(x.strip())
                for x in args.selected_gpus.split(',')
            ]
    selected_gpus_num = len(selected_gpus)

    if args.use_paddlecloud and num_nodes > 1:
        cloud_paddle_port = os.getenv("PADDLE_PORT", "")
        cloud_paddle_port_num = os.getenv("PADDLE_PORTS_NUM", "")
        if cloud_paddle_port != "" and cloud_paddle_port_num != "":
            cloud_paddle_port_num = int(cloud_paddle_port_num)
            if cloud_paddle_port_num >= selected_gpus_num:
                args.started_port = int(cloud_paddle_port)
                logger.warning("Use Cloud specified port:{}.".format(
                    cloud_paddle_port))

    trainers_endpoints = ""
    for ip in node_ips:
        for i in range(selected_gpus_num):
            if trainers_endpoints != "":
                trainers_endpoints += ","
            trainers_endpoints += "%s:%d" % (ip, args.started_port + i)

    nranks = num_nodes * selected_gpus_num

    if args.print_config:
        print("trainers_endpoints:", trainers_endpoints, ", node_id:", node_id,
              ", current_node_ip:", current_node_ip, ", num_nodes:", num_nodes,
              ", node_ips:", node_ips, ", nranks:", nranks)

    current_env = copy.copy(default_env)
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    log_fns = []
    cmds = []
    ranks = []
    for i in range(0, selected_gpus_num):
        rank = (node_id * selected_gpus_num + i)
        current_env.update({
            "FLAGS_selected_gpus": "%s" % selected_gpus[i],
            "PADDLE_TRAINER_ID": "%d" % rank,
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
        ranks.append(rank)

    try:
        alive = True
        error = False
        error_rank = []
        # wait all process finish or one error
        while alive and not error:
            alive = False
            for rank, p in zip(ranks, procs):
                ret = p.poll()
                if ret is None:
                    alive = True
                elif ret != 0:
                    error = True
                    error_rank.append(rank)
            time.sleep(1)

        if error:
            terminate_procs(procs)
            exit(1)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exit")
        terminate_procs(procs)
        raise
    except SystemExit:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_procs(procs)
        raise
    except:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_procs(procs)
        raise
    finally:
        for fn in log_fns:
            fn.close()


def launch():
    args = _parse_args()
    if args.print_config:
        _print_arguments(args)
    start_procs(args)


if __name__ == "__main__":
    launch()
