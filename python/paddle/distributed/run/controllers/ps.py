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

from .controller import Controller, ControleMode

import json


class PSController(Controller):
    @classmethod
    def enable(cls, ctx):
        if ctx.args.mode == ControleMode.PS or ctx.args.server_num or len(
                ctx.args.servers) > 0:
            ctx.logger.debug("{} enabled".format(cls.__name__))
            return True
        else:
            return False

    def build_pod(self):
        self.pod.rank = self.ctx.args.rank

        server_num = self.ctx.args.server_num or 1
        servers = [
            "{}:{}".format(self.ctx.node.ip, p)
            for p in self.ctx.node.get_free_ports(server_num)
        ]
        trainer_num = self.ctx.args.trainer_num or 1
        trainers = [
            "{}:{}".format(self.ctx.node.ip, p)
            for p in self.ctx.node.get_free_ports(trainer_num)
        ]

        data = json.dumps({
            'name': self.pod.name,
            'rank': self.pod.rank,
            'servers': servers,
            'trainers': trainers,
            'dtype': self.ctx.node.device.dtype,
        })

        peer_list, rank = self.master.sync_peers(
            '/{}/info'.format(self.job.id), self.pod.name, data,
            self.job.replicas, self.pod.rank)

        self.ctx.logger.debug("Gather peer list {}".format(peer_list))

        peer_list = [json.loads(i) for i in peer_list]

        self.save_pod_log(peer_list)

        server_endpoints = [j for i in peer_list for j in i['servers']]
        trainer_endpoints = [j for i in peer_list for j in i['trainers']]
        #rank_offset = sum([i['replicas'] for i in peer_list[:rank]])

        server_rank_offset = sum([len(i['servers']) for i in peer_list[:rank]])
        trainer_rank_offset = sum(
            [len(i['trainers']) for i in peer_list[:rank]])

        self.pod.rank = rank

        host = self.ctx.node.ip
        #self.pod.replicas = self.ctx.node.device.count

        for i in range(server_num):
            e = {
                "PADDLE_PSERVER_ENDPOINTS": ",".join(server_endpoints),
                "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints),
                "PADDLE_ROLE": "PSERVER",
                "PADDLE_RANK": "{}".format(i + server_rank_offset),
            }
            log_tag = "ps.{}".format(i)
            self.add_container(envs=e, log_tag=log_tag)

        for i in range(trainer_num):
            e = {
                "PADDLE_PSERVER_ENDPOINTS": ",".join(server_endpoints),
                "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints),
                "PADDLE_ROLE": "TRAINER_CPU",
                "PADDLE_RANK": "{}".format(i + server_rank_offset),
            }
            log_tag = "trainer.{}".format(i)
            self.add_container(envs=e, log_tag=log_tag)
