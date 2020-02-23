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
import requests
import time
import sys
from paddle.distributed.edl_utils import Edlenv
from paddle.distributed.utils import get_logger, terminate_local_procs


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
        help="The path for each process's log.If it's not setted, the log will printed to default pipe."
    )


def get_cluster():
    edl_env = edl_utils.Edlenv()
    return edl_env.get_cluster(None)


class PodProc(object):
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.cmd = None
        self.rank = None


class PodManager(object):
    def __init__(self):
        self.local_pods = {}

    def start_local_pod(job_server, job_id, pod_id, pod_rank):
        assert pod_id not in self.local_pods

        current_env = copy.copy(os.environ.copy())
        current_env.pop("http_proxy", None)
        current_env.pop("https_proxy", None)

        current_env.update({
            "PADDLE_RUNING_ENV": "PADDLE_EDL",
            "PADDLE_JOBSERVER": "%s" % edl_env.job_server,
            "PADDLE_JOB_ID": "%s" % edl_env.job_id,
            "PADDLE_POD_ID": "%d" % pod_id
        })

        logger.debug("pod proc env:{}".format(current_env))

        cmd = [sys.executable, "-u", args.training_script
               ] + args.training_script_args

        fn = None
        if args.log_dir is not None:
            os.system("mkdir -p {}".format(args.log_dir))
            fn = open("%s/pod_%s.log" % (args.log_dir, pod_id), "w")
            proc = subprocess.Popen(cmd, env=current_env, stdout=fn, stderr=fn)
        else:
            proc = subprocess.Popen(cmd, env=current_env)

        p = PodProc()
        p.proc = proc
        p.rank = pod.rank
        p.log_fn = fn
        p.cmd = cmd

        procs.append(tp)

        self.local_pods[pod_id] = p

    def kill_local_pod(pod_id):
        assert pod_id in self.local_pods
        procs = [self.local_pods[pod_id]]
        terminate_local_procs(procs)


def get_deleted_pods(cluster, cluster2):
    pods = []
    for pod in cluster.pods:
        if cluster2.get_pod_by_id(pod_id) is None:
            pods.append(pod)
    return pods


def get_added_pods(cluster, cluster2):
    pods = []
    for pod in cluster2.pods:
        if cluster.get_pod_by_id(pod_id) is None:
            pods.append(pod)
    return pods


def manage_pods():
    cluster = get_cluster()

    start_local_pod(cluster)

    while True:
        cluster2 = get_cluster()
        if cluster2 != cluster:
            for pod in get_deleted_pods():
                kill_local_pod()

            for pod in get_added_pods():
                start_local_pod()

        time.sleep(1)


if __name__ == '__main__':
    args = _parse_args()
    _print_arguments(args)

    get_logger(args.log_level)

    manage_pods()
