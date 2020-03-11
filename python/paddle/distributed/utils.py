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
import paddle.fluid as fluid
import logging
import socket
import time
import os
import signal

#loggers={}
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

    def __str__(self):
        return "job_server:{} pods:{} hdfs:{}".format(
            self.job_server, [str(pod) for pod in self.pods], self.hdfs)

    def __eq__(self, cluster):
        #print("pods length:", len(self.pods), len(cluster.pods))
        if len(self.pods) != len(cluster.pods):
            return False

        for a, b in zip(self.pods, cluster.pods):
            if a != b:
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
        self.gpu = []
        self.endpoint = None
        self.rank = None

    def __str__(self):
        return "gpu:{} endpoint:{} rank:{}".format(self.gpu, self.endpoint,
                                                   self.rank)

    def __eq__(self, t):
        if len(self.gpu) != len(t.gpu):
            return False

        if self.endpoint != t.endpoint or \
                self.rank != t.rank :
            return False

        for a, b in zip(self.gpu, t.gpu):
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

    def __str__(self):
        return "rank:{} id:{} addr:{} port:{} trainers:{}".format(
            self.rank, self.id, self.addr, self.port,
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
        for t in self.trainers:
            for g in t.gpu:
                r += "{},".format(g)

        assert r != "", "this pod {} can't see any gpus".format(self)

        r = r[:-1]
        return r


class Gloo(object):
    def __init__(self):
        self._gloo = fluid.core.Gloo()
        self._clear()

    def _clear(self):
        self._endpoints = None
        self._job_id = None
        self._hdfs = None
        self._rank = None

    def _is_changed(self, job_id, hdfs, endpoints, rank):
        return self._job_id == job_id and \
            self._hdfs == hdfs and \
            self._endpoints == endpoints and \
            self._rank == rank

    def _init(self, job_id, hdfs, endpoints, rank, try_num=3):
        if not self._is_changed(job_id, hdfs, endpoints, rank):
            self._job_id = job_id
            self._hdfs = hdfs
            self._endpoints = endpoints
            self._rank = rank
            self._try_num = try_num

            iface = self.__get_default_iface()
            try:
                self._gloo.init(rank,
                                len(self._endpoints),
                                hdfs.hdfs_path.rstrip("/") + "/edl_job_gloo",
                                hdfs.hdfs_name, hdfs.hdfs_ugi, iface,
                                self._job_id)
            except Exception as e:
                self._clear()
                logger.warning("can't init gloo")
                raise e

    def _loop(self, func, name):
        step = 0
        err = None
        while True:
            try:
                ret = func()
                return True
            except Exception as e:
                err = e
                time.sleep(3)
                step += 1
                if step >= self._try_num:
                    break

        logger.warning("gloo {} error exception:{}".format(name, err))
        return False

    def init(self, job_id, hdfs, endpoints, rank, try_num=3):
        func = functools.partial(
            self._init,
            job_id=job_id,
            hdfs=hdfs,
            endpoints=endpoints,
            rank=rank,
            try_num=try_num)
        return self._loop(func, "init")

    def barrier(self, timeout):
        #func = functools.partial(self._gloo.barrier, timeout=timeout)
        func = functools.partial(self._gloo.barrier)
        return self._loop(func, "barrier")

    def allgather(self, input, output, timeout):
        func = functools.partial(
            self._gloo.allgather, input=input, output=output, timeout=timeout)
        return self._loop(func, "allgather")

    def __get_default_iface(self):
        """
        get default physical interface
        """
        default1 = self.__get_default_iface_from_gateway()
        default2 = self.__get_default_iface_from_interfaces()
        return default2 if default1 == "lo" else default1

    def __get_default_iface_from_gateway(self):
        """
        get default physical interface
        """
        import netifaces
        gateways = netifaces.gateways()
        if gateways.get(netifaces.AF_INET) != None:
            gateway = gateways[netifaces.AF_INET]
            if len(gateway) > 0 and len(gateway[0]) > 1:
                return gateway[0][1]
        return "lo"

    def __get_default_iface_from_interfaces(self):
        """
        get default physical interface
        """
        import netifaces
        for intf_name in netifaces.interfaces():
            addresses = netifaces.ifaddresses(intf_name)
            if netifaces.AF_INET in addresses:
                ipv4_addresses = addresses[netifaces.AF_INET]
                for ipv4_address in ipv4_addresses:
                    if 'broadcast' in ipv4_address:
                        return intf_name
        return "lo"


def get_logger(log_level, name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    return logger


def get_cluster(node_ips, node_ip, paddle_port, selected_gpus):
    cluster = Cluster(hdfs=None)
    trainer_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.addr = ip
        for i in range(len(selected_gpus)):
            trainer = Trainer()
            trainer.gpu.append(selected_gpus[i])
            trainer.endpoint = "%s:%d" % (ip, paddle_port + i)
            trainer.rank = trainer_rank
            trainer_rank += 1

            pod.trainers.append(trainer)
        cluster.pods.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]


def terminate_local_procs(procs):
    for p in procs:
        if p.proc.poll() is None:
            p.proc.terminate()
            p.log_fn.close()
            logger.debug("terminate process id:{}".format(p.proc.pid))

    # wait all process terminiated
    time.sleep(3)

    for step in range(0, 10):
        alive = False
        for p in procs:
            if p.proc.poll() is None:  # not termniate
                os.kill(p.proc.pid, signal.SIGKILL)
                alive = True

        if not alive:
            logger.debug("terminate all the procs")
            return

        time.sleep(1)

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
