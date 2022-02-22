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
import os
import six
import time


class CollectiveController(Controller):
    @classmethod
    def enable(cls, ctx):
        if ctx:
            ctx.logger.debug("{} enabled".format(cls.__name__))
            return True
        else:
            return False

    def build_pod(self):
        self.pod.replicas = self.pod_replicas()

        self.pod.rank = self.ctx.args.rank if self.pod.rank < 0 else self.pod.rank

        port = self.ctx.node.get_free_port()

        data = json.dumps({
            'name': self.pod.name,
            'rank': self.pod.rank,
            'replicas': self.pod.replicas,
            'dtype': self.ctx.node.device.dtype,
            'candidate': '{}:{}'.format(self.ctx.node.ip, port)
        })

        peer_list, rank = self.master.sync_peers(
            '/{}/info'.format(self.job.id), self.pod.name, data,
            self.job.replicas, self.pod.rank)

        peer_list = [json.loads(i) for i in peer_list]

        self.ctx.logger.debug("sync peers done {}".format(peer_list))
        self.save_log(peer_list)

        global_size = sum([i['replicas'] for i in peer_list])
        rank_offset = sum([i['replicas'] for i in peer_list[:rank]])

        self.pod.rank = rank
        '''
        The new desinged collective need nothing but a master endpoint
        '''
        collective_master = peer_list[0]['candidate']

        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER": collective_master,
                "PADDLE_GLOBAL_SIZE": "{}".format(global_size),
                "PADDLE_LOCAL_SIZE": "{}".format(self.pod.replicas),
                "PADDLE_GLOBAL_RANK": "{}".format(i + rank_offset),
                "PADDLE_LOCAL_RANK": "{}".format(i),
            }
            log_file = "worker.{}.{}.log".format(self.pod.name, i)
            self.add_container(envs=e, log_file=log_file)

    '''
    compatible version of build_pod
    '''

    def _build_pod(self):

        self.pod.replicas = self.pod_replicas()

        self.pod.endpoints = [
            "{}:{}".format(self.ctx.node.ip, p)
            for p in self.ctx.node.get_free_ports(self.pod.replicas)
        ]

        eps, _ = self.master.sync_peers(
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
            c = self.init_container(envs=e)
            self.add_container(c)


class CollectiveElasticController(CollectiveController):
    @classmethod
    def enable(cls, ctx):
        if ctx.args.master.startswith("etcd://"):
            ctx.logger.debug("{} enabled".format(cls.__name__))
            return True
        else:
            return False

    def register(self):
        elastic_ttl = 6

        if self.job.id == 'default':
            self.ctx.logger.warning(
                'Using default job name may cause conflict, add --id in args')

        self.master.register_heartbeat(self.job.id, self.pod.name, elastic_ttl)

    def watch(self):
        while True:
            # self status
            status = self.pod.watch(timeout=2)

            if status == self.ctx.status.COMPLETED:
                self.master.set_status(status)
                self.pod.stop()
                return False

            elif status == self.ctx.status.FAILED:
                self.pod.stop()
                return True

            # peer status
            if self.ctx.status.need_restart() and self.master.get_status(
            ) != self.ctx.status.COMPLETED:
                return True

            #peers = self.master.fetch_peer_alive()
            #print("peers {}".format(peers))

    def run(self):

        timeout = 30
        self.register()

        while self.pod.restart <= self.ctx.args.max_restart:

            self.build_job()

            ok, replicas = self.master.wait_peer_ready(
                self.job.replicas_min, self.job.replicas_max, timeout)
            if ok:
                self.job.replicas = replicas
            else:
                break

            self.build_pod()

            self.ctx.status.run()

            assert len(self.pod.containers) > 0, "No container in the pod"
            self.ctx.logger.debug("Run pod {}\n {}".format(
                self.pod, self.pod.containers[0]))

            self.pod.deploy()

            if not self.watch():
                break

            self.pod.restart += 1
