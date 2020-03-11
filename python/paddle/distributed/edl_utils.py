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
from utils import Cluster, Pod, Trainer, logger


class Edlenv(object):
    def __init__(self):
        self.running_env = os.getenv("PADDLE_RUNING_ENV", "")
        self.job_server = os.getenv("PADDLE_JOBSERVER")
        self.job_id = os.getenv("PADDLE_JOB_ID")
        self.pod_id = os.getenv("PADDLE_POD_ID")

    def __str__(self):
        return "runing_env:{} job_server:{} job_id:{} pod_id:{}".format(
            self.running_env, self.job_server, self.job_id, self.pod_id)

    def is_under_edl(self):
        return self.running_env == "PADDLE_EDL"

    def _parse_response_pods(self, r_pods):
        t_rank = 0
        pods = []
        for rank, r_pod in enumerate(r_pods):
            #print("r_pod:", r_pod)
            pod = Pod()
            pod.rank = rank
            pod.id = r_pod["pod_id"]
            pod.addr = r_pod["addr"]
            pod.port = r_pod["pod_port"]
            pod.gpu = r_pod["gpu"]
            pod.trainers = []

            for idx, t_port in enumerate(r_pod["trainer_ports"]):
                trainer = Trainer()
                trainer.endpoint = "{}:{}".format(pod.addr, t_port)
                trainer.gpu.append(idx)
                trainer.rank = t_rank
                t_rank += 1

                pod.trainers.append(trainer)
            pods.append(pod)

        return pods

    def _get_pods_from_job_server(self):
        job_id = {'job_id': self.job_id}
        url = "{}/rest/1.0/get/query_pods".format(self.job_server)
        logger.debug("query pods from url:{}".format(url))

        step = 0
        pods = {}
        while True:
            try:
                r = requests.get(url, params=job_id)
                d = r.json()
                assert self.job_id == d["job_id"], "job_id is not {}".format(
                    self.job_id)
                pods = d["pods"]
                logger.debug("job_server:{} response:{}".format(r.url, pods))
                break
            except Exception as e:
                step += 1
                if step > 10:
                    logger.error(
                        "get pods from job_server:{} error, try again!".format(
                            url))
                    sys.exit(1)

                logger.warning(
                    "get pods from job_server:{} payload:{} error:{}, try again!".
                    format(url, job_id, str(e)))
                time.sleep(3)

        return self._parse_response_pods(pods)

    def get_cluster(self, hdfs):
        assert self.is_under_edl(), "Edlenv only used under edl environments"

        pods = self._get_pods_from_job_server()

        cluster = Cluster(hdfs)
        cluster.job_server = self.job_server
        cluster.job_id = self.job_id
        cluster.pods = pods
        cluster.hdfs = hdfs

        logger.debug("get clsuter:{} from jobserver edl_env:{}".format(cluster,
                                                                       self))

        pod = cluster.get_pod_by_id(self.pod_id)
        return cluster, pod


def barrier_terminate_world_trainers(cluster, pod, comm, timeout=10, try_num=1):
    step = 0
    r = comm.init(
        job_id=cluster.job_id,
        hdfs=cluster.hdfs,
        endpoints=cluster.pods_endpoints(),
        rank=pod.rank,
        try_num=try_num)
    if not r:
        logger.warning("can't init from context:{}".format(cluster))
        return False

    r = comm.barrier(timeout)
    if not r:
        logger.warning("barrier timeout context:{}".format(cluster))
        return False

    return True
