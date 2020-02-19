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

logger = None


class Hdfs(object):
    def __init__(self):
        self.hdfs_ugi = None
        self.hdfs_name = None
        self.hdfs_path = None


class Cluster(object):
    def __init__(self):
        self.job_server = None
        self.pods = None
        self.hdfs = None

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return True

        for pod in self.pods:
            if pod != cluster.pods[i]:
                return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self.pods = copy.copy(cluster.pods)

    def trainer_nranks(self):
        count = 0
        for pod in self.pods:
            for gpu in pod.gpus:
                count += 1
        return count

    def pod_nranks(self):
        return len(self.pods)

    def get_trainers_endpoints(self):
        r = []
        for pod in self.pods:
            for t in pod.trainers:
                r.append(t.endpoint)
        return r

    def get_pods_endpints(self):
        r = []
        for pod in self.pods:
            ep = "{}:{}".format(pod.ip, pod.port)
            assert pod.port != None and pod.ip != None, "{} not a valid endpoint".format(
                ep)
            r.append(ep)

        return r

    def get_pod(self, pod_id):
        for pod in self.pods:
            if pod_id == pod.id:
                return pod

        return None


class JobServer(object):
    def __init__(self):
        self.endpoint = None


class Trainer(object):
    def __init__(self):
        self.gpu = []
        self.endpoint = None
        self.rank = None

    def __eq__(self):
        pass

    def __ne__(self):
        pass

    def rank(self):
        return self.rank


class Pod(object):
    def __init__(self):
        self.rank = None  # pod_id
        self.id = None  # node rank
        self.addr = None
        self.port = None
        self.trainers = []

    def __eq__(self, pod):
        if self.ranks != pod.rank or \
                self.id != pod.id or \
                self.addr != pod.addr or \
                self.port != pod.port:
            return False

        if len(self.trainers) != pod.trainers:
            return False

        for i in range(len(self.trainers)):
            if self.trainers[i] != pod.trainers[i]:
                return False

        return True

    def __ne__(self, pod):
        return not self == pod

    def parse_response(self, res_pods):
        pass

    def rank(self):
        return self.rank


class Gloo(object):
    def __init__():
        self._prefix = "edl_job"
        self._gloo = fluid.core.Gloo()

        self._endpoints = None
        self._hdfs = None
        self._rank = None

    def _clear():
        self._endpoints = None
        self._hdfs = None
        self._rank = None

    def _init(hdfs, endpints, rank, try_num=3):
        if endpoints != self._endpoints or hdfs != self._hdfs or rank != self._rank:
            self._hdfs = hdfs
            self._endpoints = self._endpoints
            self._rank = rank
            self._try_num = try_num

            iface = self.__get_default_iface()
            if not self.__gloo.init(pod.idx,
                                    len(pods_endpoints),
                                    hdfs.hdfs_path.rstrip("/") + "/all",
                                    hdfs.hdfs_name, hdfs.hdfs_ugi, self.__iface,
                                    self._prefix):
                self._clear()
                return False

        return True

    def _loop(func):
        while True:
            if func():
                return True

            if step > self._try_num:
                break

            time.sleep(3)
            step += 1

        return False

    def init(hdfs, endpoints, rank, try_num=3):
        func = functools.partial(
            self._init,
            hdfs=hdfs,
            endpoints=endpoints,
            rank=rank,
            try_num=try_num)
        return self._loop(func)

    def barrier(timeout):
        func = functools.partial(self._gloo.barrier, timeout=timeout)
        return self._loop(func)

    def allgather(input, output, timeout):
        func = functools.partial(
            self._gloo.allgather, input=input, output=output, timeout=timeout)
        return self._loop(func)

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


def get_logger(log_level):
    l = logging.getLogger()

    # initial log with loglevel
    l.setLevel(log_level)
    log_handler = logging.StreamHandler()
    log_format = logging.Formatter(
        '%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    log_handler.setFormatter(log_format)
    l.addHandler(log_handler)

    return l


def get_cluster(node_ips, node_ip, started_port, selected_gpus):
    cluster = Cluster()
    trainer_rank = 0
    for node_rank, ip in enumerate(node_ips):
        pod = Pod()
        pod.rank = node_rank
        pod.ip = ip
        for gpu in range(len(selected_gpus)):
            trainer = Trainer()
            trainer.gpu = gpu
            trainer.endpoint = "%s:%d" % (ip, paddle_port + i)
            trainer.rank = trainer_rank
            trainer_rank += 1

            pod.trainers.append(trainer)
        cluster.pod.append(pod)

    pod_rank = node_ips.index(node_ip)
    return cluster, cluster.pods[pod_rank]
