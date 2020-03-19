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
            pod = Pod()
            pod.rank = rank
            pod.id = r_pod["pod_id"]
            pod.addr = r_pod["addr"]
            pod.port = r_pod["pod_port"]
            pod.gpus = r_pod["gpus"]
            pod.trainers = []

            for idx, t_port in enumerate(r_pod["trainer_ports"]):
                trainer = Trainer()
                trainer.gpus = []
                trainer.endpoint = "{}:{}".format(pod.addr, t_port)
                trainer.rank = t_rank
                trainer.gpus = [idx]
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
                flag = d["job_stage_flag"]
                logger.debug("job_server:{} response:{}".format(r.url, pods))
                break
            except Exception as e:
                step += 1
                if step >= 30 * 60 / 3:  # 30 minutes
                    logger.error(
                        "get pods from job_server:{} error, try again!".format(
                            url))
                    sys.exit(1)
                logger.warning(
                    "get pods from job_server:{} payload:{} error:{}, try again!".
                    format(url, job_id, str(e)))
                time.sleep(3)

        return self._parse_response_pods(pods), flag

    def get_cluster(self, hdfs):
        assert self.is_under_edl(), "Edlenv only used under edl environments"

        pods, flag = self._get_pods_from_job_server()

        cluster = Cluster(hdfs)
        cluster.job_server = self.job_server
        cluster.job_id = self.job_id
        cluster.pods = pods
        cluster.hdfs = hdfs
        cluster.job_stage_flag = flag

        logger.debug("get clsuter:{} from jobserver edl_env:{}".format(cluster,
                                                                       self))

        pod = cluster.get_pod_by_id(self.pod_id)
        return cluster, pod


def _post_kv(url, scope, key, value):
    kv = {"scope": scope, "key": key, "value": value}
    url = "http://{}/rest/1.0/post/pod".format(url)
    print("post:", kv)
    try:
        r = requests.get(url, params=kv)
        d = r.json()
        logger.info("url:{} response:{}".format(r.url, d))
        return True
    except Exception as e:
        logger.fatal("url:{}, error:{}".format(url, e))
    return False


def _get_scope(url, scope):
    kv = {"scope": scope}
    url = "http://{}/rest/1.0/get/scope".format(url)
    try:
        r = requests.get(url, params=kv)
        d = r.json()
        logger.info("url:{} response:{}".format(r.url, d))
        return d["value"]
    except Exception as e:
        logger.warning("url:{}, error:{}".format(url, e))
        return None


def _is_scope_full(cluster, scope_kvs):
    for pod in cluster.pods:
        if pod.id not in scope_kvs:
            return False
    return True


def barrier(cluster, pod):
    # check to start a httpserver
    # post job_stage_flag,pod_id,pod_id to httpstore
    url = "{}:{}".format(cluster.pods[0].addr, cluster.pods[0].port)
    if not _post_kv(url, cluster.job_stage_flag, pod.id, pod.id):
        return False

    scope_kvs = _get_scope(url, cluster.job_stage_flag)
    if scope_kvs is None:
        return False

    if not _is_scope_full(cluster, scope_kvs):
        return False

    return True
