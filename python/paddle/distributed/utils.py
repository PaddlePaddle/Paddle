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

import functools
import logging
import socket
import time
import os
import signal
import copy
import sys
import subprocess
from contextlib import closing
import socket

logger = logging.getLogger("root")
logger.propagate = False


class Hdfs(object):
    def __init__(self):
        self.hdfs_ugi = None
        self.hdfs_name = None
        self.hdfs_path = None

    def is_valid(self):
        return self.hdfs_ugi is not None and \
            self.hdfs_name is not None and \
            self.hdfs_path is not None

    def __str__(self):
        return "hdfs_ugi:{} hdfs_name:{} hdfs_path{}".format(
            self.hdfs_ugi, self.hdfs_name, self.hdfs_path)

    def __eq__(self, n):
        return self.hdfs_ugi == n.hdfs_ugi and \
            self.hdfs_name == n.hdfs_name and \
            self.hdfs_path == n.hdfs_path

    def __ne__(self, n):
        return not self == n


class Cluster(object):
    def __init__(self, hdfs):
        self.job_server = None
        self.pods = []
        self.hdfs = None
        self.job_stage_flag = None

    def __str__(self):
        return "job_server:{} pods:{} job_stage_flag:{} hdfs:{}".format(
            self.job_server, [str(pod) for pod in self.pods],
            self.job_stage_flag, self.hdfs)

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return False

        for a, b in zip(self.pods, cluster.pods):
            if a != b:
                return False

        if self.job_stage_flag != cluster.job_stage_flag:
            return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self.pods = copy.copy(cluster.pods)

    def trainers_nranks(self):
        return len(self.trainers_endpoints())

    def pods_nranks(self):
        return len(self.pods)

    def trainers_endpoints(self):
        r = []
        for pod in self.pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def pods_endpoints(self):
        r = []
        for pod in self.pods:
            ep = "{}:{}".format(pod.addr, pod.port)
            assert pod.port != None and pod.addr != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod_by_id(self, pod_id):
        for pod in self.pods:
            if str(pod_id) == str(pod.id):
                return pod

        return None


class JobServer(object):
    def __init__(self):
        self.endpoint = None

    def __str__(self):
        return "{}".format(self.endpoint)

    def __eq__(self, j):
        return self.endpint == j.endpoint

    def __ne__(self, j):
        return not self == j


class Trainer(object):
    def __init__(self):
        self.gpus = []
        self.endpoint = None
        self.rank = None

    def __str__(self):
        return "gpu:{} endpoint:{} rank:{}".format(self.gpus, self.endpoint,
                                                   self.rank)

    def __eq__(self, t):
        if len(self.gpus) != len(t.gpus):
            return False

        if self.endpoint != t.endpoint or \
                self.rank != t.rank :
            return False

        for a, b in zip(self.gpus, t.gpus):
            if a != b:
                return False

        return True

    def __ne__(self, t):
        return not self == t

    def rank(self):
        return self.rank


class Pod(object):
    def __init__(self):
        self.rank = None
        self.id = None
        self.addr = None
        self.port = None
        self.trainers = []
        self.gpus = []

    def __str__(self):
        return "rank:{} id:{} addr:{} port:{} visible_gpu:{} trainers:{}".format(
            self.rank, self.id, self.addr, self.port, self.gpus,
            [str(t) for t in self.trainers])

    def __eq__(self, pod):
        if self.rank != pod.rank or \
                self.id != pod.id or \
                self.addr != pod.addr or \
                self.port != pod.port:
            logger.debug("pod {} != pod".format(self, pod))
            return False

        if len(self.trainers) != len(pod.trainers):
            logger.debug("trainers {} != {}".format(self.trainers,
                                                    pod.trainers))
            return False

        for i in range(len(self.trainers)):
            if self.trainers[i] != pod.trainers[i]:
                logger.debug("trainer {} != {}".format(self.trainers[i],
                                                       pod.trainers[i]))
                return False

        return True

    def __ne__(self, pod):
        return not self == pod

    def parse_response(self, res_pods):
        pass

    def rank(self):
        return self.rank

    def get_visible_gpus(self):
        r = ""
        for g in self.gpus:
            r += "{},".format(g)

        assert r != "", "this pod {} can't see any gpus".format(self)

        r = r[:-1]
        return r


def get_logger(log_level, name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


def get_cluster(node_ips, node_ip, paddle_ports, selected_gpus):
    assert type(paddle_ports) is list, "paddle_ports must be list"
    cluster = Cluster(hdfs=None)
    trainer_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        for i in range(len(selected_gpus)):
            trainer = Trainer()
            trainer.gpus.append(selected_gpus[i])
            trainer.endpoint = "%s:%d" % (ip, paddle_ports[i])
            trainer.rank = trainer_rank
            trainer_rank += 1

            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def terminate_local_procs(procs):
    for p in procs:
        if p.proc.poll() is None:
            # subprocess need to release resource(e.g. shared memory)
            # use join to wait subprocess releasing
            p.proc.join(timeout=1)
            p.log_fn.close()
            logger.debug("terminate process id:{}".format(p.proc.pid))

    # wait all process terminiated
    # time.sleep(3)

    for step in range(0, 50):
        alive = False
        for p in procs:
            if p.proc.poll() is None:  # not termniate
                os.kill(p.proc.pid, signal.SIGKILL)
                alive = True

        if not alive:
            logger.info("terminate all the procs")
            return

        time.sleep(3)

    logger.fatal("can't kill all process and exit")
    exit(1)


def get_host_name_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_name, host_ip
    except:
        return None


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def find_free_ports(num):
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    port_set = set()
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set.add(port)

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 100:
            print(
                "can't find avilable port and use the specified static port now!"
            )
            return None

    return None


class TrainerProc(object):
    def __init__(self):
        self.proc = None
        self.log_fn = None
        self.rank = None
        self.cmd = None


def start_local_trainers(cluster,
                         pod,
                         training_script,
                         training_script_args,
                         log_dir=None):
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

        cmd = [sys.executable, "-u", training_script] + training_script_args

        logger.info("start trainer proc:{} env:{}".format(cmd, proc_env))

        fn = None
        if log_dir is not None:
            os.system("mkdir -p {}".format(log_dir))
            fn = open("%s/workerlog.%d" % (log_dir, idx), "a")
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
    try:
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
