# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
from paddle.distributed.edl_utils import Edlenv
from paddle.distributed.utils import logger, get_logger, terminate_local_procs, get_host_name_ip
from argparse import ArgumentParser, REMAINDER
import six
import copy
import subprocess
import time


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
    parser = ArgumentParser(description='''start job_client_demo''')

    parser.add_argument(
        "--log_level",
        type=int,
        default=20,  # logging.INFO, details are here:https://docs.python.org/3/library/logging.html#levels
        help="Logging level, default is logging.INFO")

    parser.add_argument(
        "--log_dir",
        type=str,
        default="pod_log",
        help="The path for each pod's log.")

    parser.add_argument(
        "--package_sh",
        type=str,
        default=None,
        help="The bash shell to make pod env.")

    parser.add_argument(
        "--pod_path", type=str, help="The pod shell path to execute.")

    #positional
    parser.add_argument(
        "training_script", type=str, help="The full path to start trainer proc")

    #rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def get_cluster():
    edl_env = Edlenv()
    cluster, _ = edl_env.get_cluster(None)
    return cluster


class PodProc(object):
    def __init__(self):
        self.env = None
        self.proc = None
        self.log_fn = None
        self.cmd = None
        self.rank = None

    def __str__(self):
        return "env:{} proc:{} log_fn:{} cmd:{} rank:{}".format(
            self.env, self.proc.pid, self.log_fn, self.cmd, self.rank)


class PodManager(object):
    def __init__(self):
        self.local_pods = {}

    def start_local_pods(self, cluster):
        host_name, host_ip = get_host_name_ip()
        #gpu_rank = 0
        for pod in cluster.pods:
            print("pod.addr:", pod.addr, "host_name:", host_name, "host_ip:",
                  host_ip)
            if pod.addr == "127.0.0.1" or \
                    pod.addr==host_name or \
                    pod.addr == host_ip:
                self.start_local_pod(cluster.job_server, cluster.job_id, pod)
                #gpu_rank += 1

    def start_local_pod(self, job_server, job_id, pod):
        assert pod.id not in self.local_pods, "pod_id:{} local_pods:{}".format(
            pod.id, [k for k, _ in self.local_pods.items()].sort())

        if args.package_sh is not None:
            cmd = args.package_sh + " -pod_id {}".format(pod.id)
            print("execute cmd:", cmd)
            ret = os.system(cmd)
            assert ret == 0, "execute {} error!".format(cmd)

        current_env = copy.copy(os.environ.copy())
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        pod_env = ({
            "PADDLE_RUNING_ENV": "PADDLE_EDL",
            "PADDLE_JOBSERVER": "%s" % job_server,
            "PADDLE_JOB_ID": "%s" % job_id,
            "PADDLE_POD_ID": "%s" % pod.id,
            "CUDA_VISIBLE_DEVICES": "%s" % pod.get_visible_gpus()
        })

        current_env.update(pod_env)

        fn = None
        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/pod_%s.log" % (args.log_dir, pod.id), "w")

        wd = os.getcwd()
        if args.pod_path is not None:
            pod_path = args.pod_path + "/{}".format(pod.id)
            os.chdir(pod_path)

        #cmd = [sys.executable, "-u", args.training_script
        #       ] + args.training_script_args
        cmd = ["bash", args.training_script]

        logger.info("start pod proc env:{} cmd:{}".format(pod_env, cmd))

        if args.log_dir is not None:
            #os.system("mkdir -p {}".format(args.log_dir))
            #fn = open("%s/pod_%s.log" % (args.log_dir, pod.id), "w")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        os.chdir(wd)

        p = PodProc()
        p.proc = proc
        p.rank = pod.rank
        p.log_fn = fn
        p.cmd = cmd
        p.env = pod_env

        self.local_pods[pod.id] = p

    def kill_local_pod(self, pod_id):
        if pod_id not in self.local_pods:
            return
            #"pod_id:{} local_pods:{}".format(pod_id, [k for k, _ in self.local_pods.items()])

        procs = [self.local_pods[pod_id]]
        logger.info("kill pod_id:{} pod:{}".format(
            pod_id, [str(proc) for proc in procs]))
        terminate_local_procs(procs)

        #remove from local_pods
        self.local_pods.pop(pod_id)


def get_deleted_pods(cluster, cluster2):
    pods = []
    for pod in cluster.pods:
        if cluster2.get_pod_by_id(pod.id) is None:
            pods.append(pod)
    return pods


def get_added_pods(cluster, cluster2):
    pods = []
    for pod in cluster2.pods:
        if cluster.get_pod_by_id(pod.id) is None:
            pods.append(pod)
    return pods


def manage_pods():
    cluster = get_cluster()
    logger.debug("get_cluster:", cluster)

    pod_manager = PodManager()
    pod_manager.start_local_pods(cluster)

    while True:
        cluster2 = get_cluster()
        if cluster2 != cluster:
            deleted_pods = get_deleted_pods(cluster, cluster2)
            logger.info("deleted_pods:", deleted_pods)
            for pod in deleted_pods:
                pod_manager.kill_local_pod(pod.id)

            added_pods = get_added_pods(cluster, cluster2)
            logger.info("added_pods:", added_pods)
            #gpu_rank = 0
            for pod in added_pods:
                pod_manager.start_local_pod(cluster2.job_server,
                                            cluster2.job_id, pod)
                #gpu_rank += 1

            cluster = cluster2

        time.sleep(3)


if __name__ == '__main__':
    args = _parse_args()
    _print_arguments(args)

    get_logger(args.log_level)

    manage_pods()
