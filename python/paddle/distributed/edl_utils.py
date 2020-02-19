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
import utils


class Edlenv(object):
    def __init__(self):
        self.running_env = os.getenv("PADDLE_RUNING_ENV", "")
        self.job_server = os.getenv("PADDLE_JOBSERVER")
        self.job_id = os.getenv("PADDLE_JOB_ID")
        self.pod_id = os.getenv("PADDLE_POD_ID")

    def is_under_edl(self):
        return self.running_env == "PADDLE_EDL"

    def _parse_response_pods(r_pods):
        for rank, r_pod in enumerate(r_pods):
            pod = Pod()
            pod.rank = rank
            pod.id = r_pod.pod_id
            pod.addr = r_pod.addr
            pod.port = r_pod.pod_port

            for idx, t_port in enumerate(r_pod.trainer_ports):
                trainer = Trainer()
                trainer.endpoint = pod.addr + ":" + t_port
                trainer.gpu.append(idx)
                pod.trainers.append(trainer)

        return pods

    def _get_pods_from_job_server(self):
        job_id = {'job_id': self.job_id}
        url = "{}/rest/1.0/get/query_pods".format(self.job_server)

        step = 0
        pods = {}
        while True:
            try:
                r = requests.get(url, params=job_id)
                d = r.json()
                pods = d["job_id"]
                logger.debug("get pods:{} from job_server:{}".format(
                    pods, r.job_server))
                break
            except:
                step += 1
                if step > 10:
                    logger.error(
                        "get pods from job_server:{} error, try again!".format(
                            r.url))
                    sys.exit(1)

                logger.warning("get pods from job_server:{} error, try again!".
                               format(r.url))
                time.sleep(3)

        return self._parse_response_pods(pods)

    def get_cluster(self):
        assert self.is_under_edl(), "Edlenv only used under edl environments"

        cluster = utils.Cluster()
        cluster.job_server = self.job_server
        cluster.job_id = self.job_id
        cluster.pods = pods

        return cluster


def barrier_terminate_world_trainers(cluster, pod, comm, timeout=10, try_num=3):
    pods_endpoints = cluster.get_pods_endpoints()

    step = 0
    while True:
        r = comm.init(cluster.hdfs, pods_endpoints, pods.idx, try_num=try_num)
        if not r:
            return False

        r = comm.barrier(timeout)
        if not r:
            return False

    return False
