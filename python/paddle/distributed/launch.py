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
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle.fluid as fluid

from utils import *
import cloud_util as cloud
import edl_util as edl


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
        help="It's for gpu trainning and the trainning process will run on the selected_gpus,"
        "each process is bound to a single GPU. And if it's not setted, this module will use all the gpu cards for training."
    )

    parser.add_argument(
        "--log_level",
        type=int,
        default=20,  # logging.INFO, details are here:https://docs.python.org/3/library/logging.html#levels
        help="Logging level, default is logging.INFO")

    parser.add_argument(
        "--log_dir",
        type=str,
        help="The path for each process's log.If it's not setted, the log will printed to default pipe."
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


def terminate_local_trainers(procs):
    for p in procs:
        if p.proc.poll() is None:
            p.terminate()
            p.log_fn.close()

    # wait all process terminiated
    time.sleep(5)

    alive = False
    for step in range(0, 100):
        for p in procs:
            if p.proc.poll() is not None:
                os.kill(p.pid, SIGKILL)
                alive = True
        if not alive:
            return

        time.sleep(10)

    print("can't kill all process and exit")
    exit(1)


def get_gpus():
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

    return selected_gpus


class TrainerProc(object):
    def __init__():
        self.proc = None
        self.log_fn = None
        self.rank = None
        self.cmd = None


def start_local_trainers(pod):
    current_env = copy.copy(os.environ.copy())
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for t in len(pod.trainers):
        current_env.update({
            "FLAGS_selected_gpus": "%s" % t.gpu,
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s:%d" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.world_ranks(),
            "PADDLE_TRAINER_ENDPOINTS": cluster.trainer_endpoints
        })

        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args

        fn = None
        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/workerlog.%d" % (args.log_dir, i), "w")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


def watch_local_trainers(procs):
    try:
        alive = True
        error = False
        error_rank = []
        # wait all process finish or one error
        alive = False
        for p in procs:
            ret = p.proc.poll()
            if ret is None:
                alive = True
            elif ret != 0:
                error = True
                error_rank.append(p.rank)

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

    return alive


def get_cluster_from_args(args, selected_gpus):
    node_ips = [x.strip() for x in args.cluster_node_ips.split(',')]
    node_ip = args.node_ip
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args:node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    return get_cluster(node_ips, node_ip, args.started_port, selected_gpus)


def launch(args):
    # parse arguments, used for cloud-single-machine and local
    selected_gpus = get_gpus()
    trainer_nums = cloud.get_trainers_num()
    logger.debug("parsed from args trainerss_num:{} selected_gpus:{}".format(
        trainers_num, selected_gpus))

    cluster = None
    comm = None
    use_edl = edl.is_under_edl()
    if args.use_paddlecloud and not use_edl and trainers_num != 1:
        cluster = cloud.get_cloud_cluster(arges.node_ips, args.node_ip,
                                          args.started_port, selected_gpus)
        logger.info("get cluster from cloud:{}".format(cluster))
    elif use_edl:
        edl_env = Edlenv()
        cluster = edl_env.get_cluster()
        comm = Gloo()
        logger.info("get cluster from edl:{}".format(cluster))
    else:
        cluster = get_cluster_from_args(args, selected_gpus)
        logger.info("get cluster from args:{}".format(cluster))

    procs = start_local_trainers(cluster, pod)
    step = 0
    while True:
        if use_edl:
            cluster2 = edl_env.get_cluster()
            pod = cluster2.pod(edl_env.pod_id)
            if pod is None:  # me is dead
                logger.info(
                    "Cluster changed. This pod is not exist so exit(0)! \
                    New cluster:{}. Old Cluster:{}".format(cluster2, cluster))
                sys.exit(0)

            if cluster2 != cluster:
                logger.info("Cluster changed. New cluster:{}. Old Cluster:{}".
                            format(cluster2, cluster))
                terminate_local_trainers(procs)

                if not barrier_terminate_world_trainers(cluster, comm):
                    logger.warning("Can't barrier in cluster:{}".format(
                        cluster))
                    continue

                procs = start_local_trainers(cluster, pod)

        alive = watch_local_trainers(procs)

        if not alive:
            logger.info("Local procs complete, POD info:{}".format(pod))
            return

        time.sleep(1)


if __name__ == "__main__":
    get_logger()

    args = _parse_args()
    if args.print_config:
        _print_arguments(args)

    launch(args)
