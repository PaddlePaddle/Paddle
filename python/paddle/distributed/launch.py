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

import cloud_utils
import edl_utils
from http_store import kv_server


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

    parser.add_argument(
        "--hdfs_name",
        type=str,
        default=None,
        help="The hdfs_name used for edl.")

    parser.add_argument(
        "--hdfs_ugi", type=str, default=None, help="The hdfs_ugi used for edl.")

    # checkpoint will saved here
    parser.add_argument(
        "--hdfs_path",
        type=str,
        default=None,
        help="The hdfs_path used for edl.")
    """
    #checkpoint path
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoint",
        help="The hdfs_path used for edl.")
    """

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
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.rank = None
        self.cmd = None


def start_local_trainers(cluster, pod):
    current_env = copy.copy(os.environ.copy())
    #paddle broadcast ncclUniqueId use socket, and
    #proxy maybe make trainers unreachable, so delete them.
    #if we set them to "", grpc will log error message "bad uri"
    #so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for idx, t in enumerate(pod.trainers):
        proc_env = {
            "FLAGS_selected_gpus": "%s" % ",".join([str(g) for g in t.gpus]),
            "PADDLE_TRAINER_ID": "%d" % t.rank,
            "PADDLE_CURRENT_ENDPOINT": "%s" % t.endpoint,
            "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints())
        }

        current_env.update(proc_env)

        logger.debug("trainer proc env:{}".format(current_env))

        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args

        logger.info("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None
        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/workerlog.%d" % (args.log_dir, idx), "a")
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


def watch_local_trainers(procs, nranks):
    #print("procs num:", len(procs))
    try:
        error = False
        error_rank = []
        # wait all process finish or one error
        alive = False
        for p in procs:
            ret = p.proc.poll()
            #print("proc poll ret:", ret)
            if ret is None:
                alive = True
            elif ret != 0:
                error = True
                error_rank.append(p.rank)

        if error:
            terminate_local_procs(procs)
            exit(1)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt, exit")
        terminate_local_procs(procs)
        raise
    except SystemExit:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise
    except:
        logger.error(
            "ABORT!!! Out of all {} trainers, the trainer process with rank={} was aborted. Please check its log.".
            format(nranks, error_rank))
        terminate_local_procs(procs)
        raise

    return alive


def get_cluster_from_args(args, selected_gpus):
    node_ips = [x.strip() for x in args.cluster_node_ips.split(',')]
    node_ip = args.node_ip
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args:node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    return get_cluster(node_ips, node_ip, args.started_port, selected_gpus)


def get_hdfs_from_args(args):
    hdfs = Hdfs()
    hdfs.hdfs_name = args.hdfs_name
    hdfs.hdfs_ugi = args.hdfs_ugi
    hdfs.hdfs_path = args.hdfs_path

    #assert hdfs.is_valid()
    return hdfs


def edl_barrier(edl_env, hdfs, timeout=-1):
    cluster = None
    pod = None
    step = 0
    while True:
        cluster, pod = edl_env.get_cluster(hdfs)
        if pod is None:  # me is dead
            logger.info("This pod is not exist so exit(0)! Cluster:{} pod:{}".
                        format(cluster, pod))
            sys.exit(0)

        if pod.rank == 0 and not kv_server.is_alive():
            kv_server.start("0.0.0.0", pod.port)

        ret = edl_utils.barrier(cluster=cluster, pod=pod)
        if ret:
            break

        logger.warning("Can't barrier in cluster:{}:{}".format(pod.addr,
                                                               pod.port))
        time.sleep(3)
        if timeout > 0 and step >= timeout:
            logger.warning("can't barrier to start now!so exit")
            sys.exit(1)
        continue

    logger.info("edl_barrier_start ok!")
    return cluster, pod


def launch(args):
    # parse arguments, used for cloud-single-machine and local
    selected_gpus = get_gpus()
    trainers_num = cloud_utils.get_trainers_num()
    logger.debug("parsed from args trainerss_num:{} selected_gpus:{}".format(
        trainers_num, selected_gpus))

    cluster = None
    pod = None
    #comm = None
    hdfs = None

    edl_env = edl_utils.Edlenv()
    use_edl = edl_env.is_under_edl()

    if args.use_paddlecloud and not use_edl and trainers_num != 1:
        cluster, pod = cloud_utils.get_cloud_cluster(
            arges.node_ips, args.node_ip, args.started_port, selected_gpus)
        logger.info("get cluster from cloud:{}".format(cluster))
    elif use_edl:
        hdfs = get_hdfs_from_args(args)
        cluster, pod = edl_barrier(edl_env, hdfs, timeout=15 * 60)
        logger.info("get cluster from edl:{}".format(cluster))
    else:
        cluster, pod = get_cluster_from_args(args, selected_gpus)
        logger.info("get cluster from args:{}".format(cluster))

    if use_edl:
        procs = start_local_trainers(cluster, pod)
    else:
        procs = start_local_trainers(cluster, pod)

    #try_step=0
    while True:
        if use_edl:
            cluster2, pod = edl_env.get_cluster(hdfs)

            if cluster2 != cluster:
                logger.info("Cluster changed. New cluster:{}. Old Cluster:{}".
                            format(cluster2, cluster))
                terminate_local_procs(procs)

                cluster, pod = edl_barrier(edl_env, hdfs, timeout=30 * 60)

                procs = start_local_trainers(cluster, pod)

        alive = watch_local_trainers(procs, cluster.trainers_nranks())

        if not alive:
            logger.info("Local procs complete, POD info:{}".format(pod))
            break

        time.sleep(3)

    edl_barrier(edl_env, hdfs)


if __name__ == "__main__":
    args = _parse_args()

    logger = get_logger(args.log_level)

    if args.print_config:
        _print_arguments(args)

    launch(args)
