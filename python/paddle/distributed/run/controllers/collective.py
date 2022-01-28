# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .controller import Controller

import json


class CollectiveController(Controller):
    @classmethod
    def enable(cls, ctx):
        ctx.logger.debug("CollectiveController enabled")
        if ctx:
            return True
        else:
            return False

    def build_job(self):

        self.job.replicas = self.ctx.args.np or 1

    def build_pod(self):
        self.pod.replicas = self.pod_replicas()

        data = json.dumps({
            'name': self.pod.name,
            'rank': self.pod.rank,
            'replicas': self.pod.replicas,
            'dtype': self.ctx.node.device.dtype,
        })

        peer_list, rank = self.store.allgather(
            '/info',
            self.pod.name,
            data,
            self.job.replicas, )

        peer_list = [json.loads(i) for i in peer_list]

        global_size = sum([i['replicas'] for i in peer_list])
        rank_offset = sum([i['replicas'] for i in peer_list[:rank]])

        self.pod.rank = rank

        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER": self.store.master,
                "PADDLE_GLOBAL_SIZE": "{}".format(global_size),
                "PADDLE_LOCAL_SIZE": "{}".format(self.pod.replicas),
                "PADDLE_GLOBAL_RANK": "{}".format(i + rank_offset),
                "PADDLE_LOCAL_RANK": "{}".format(i),
            }
            self.add_container(envs=e)

    '''
    compatible version of build_pod
    '''

    def _build_pod(self):

        self.pod.replicas = self.pod_replicas()

        self.pod.endpoints = [
            "{}:{}".format(self.ctx.node.ip, p)
            for p in self.ctx.node.get_free_ports(self.pod.replicas)
        ]

        eps, _ = self.store.allgather(
            '/workers',
            self.pod.name,
            ",".join(self.pod.endpoints),
            self.job.replicas, )

        self.job.endpoints = ",".join(eps).split(",")

        for i in range(self.pod.replicas):
            e = {
                "PADDLE_TRAINER_ENDPOINTS": ",".join(self.job.endpoints),
                "PADDLE_CURRENT_ENDPOINT": self.pod.endpoints[i],
                "PADDLE_TRAINER_ID":
                "%d" % self.job.endpoints.index(self.pod.endpoints[i]),
                "PADDLE_TRAINERS_NUM": "%d" % len(self.job.endpoints),
                "PADDLE_RANK_IN_NODE": str(i),
            }
            c = self.build_container(envs=e)
            self.add_container(c)
