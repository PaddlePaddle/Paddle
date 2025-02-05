# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import shutil

from .controller import Controller, ControllerMode


class PSController(Controller):
    @classmethod
    def enable(cls, ctx):
        if (
            ctx.args.run_mode == ControllerMode.PS
            or ctx.args.server_num
            or len(ctx.args.servers) > 0
            or ctx.args.trainer_num
            or len(ctx.args.trainers) > 0
        ):
            ctx.logger.debug(f"{cls.__name__} enabled")
            ctx.args.run_mode = ControllerMode.PS
            return True
        else:
            return False

    def build_pod(self):
        if self.ctx.args.servers and self.ctx.args.trainers:
            self._build_pod_with_args()
        else:
            self._build_pod_with_master()

    def _build_pod_with_args(self):
        if '127.0.0.1' in self.ctx.args.servers:
            host = '127.0.0.1'
        else:
            host = self.ctx.node.ip

        server_endpoints = list(self.ctx.args.servers.split(","))
        trainer_endpoints = list(self.ctx.args.trainers.split(","))
        servers = [
            s for s in self.ctx.args.servers.split(",") if s.startswith(host)
        ]
        trainers = [
            s for s in self.ctx.args.trainers.split(",") if s.startswith(host)
        ]
        server_num = len(servers)
        trainer_num = len(trainers)

        self.pod.replicas = server_num + trainer_num

        self.save_pod_log([server_endpoints, trainer_endpoints])

        import tempfile

        gloo_rendezvous_dir = tempfile.mkdtemp()
        if os.path.exists(gloo_rendezvous_dir):
            shutil.rmtree(gloo_rendezvous_dir)

        gloo_port = self.ctx.args.gloo_port
        gloo_http = "{}:{}".format(server_endpoints[0].split(":")[0], gloo_port)

        _gloo_envs = {
            "PADDLE_GLOO_RENDEZVOUS": "3",
            "PADDLE_GLOO_FS_PATH": gloo_rendezvous_dir,
            "PADDLE_GLOO_HTTP_ENDPOINT": gloo_http,
            "PADDLE_WITH_GLOO": self.ctx.args.with_gloo,
        }

        for i in range(server_num):
            e = {
                "PADDLE_PSERVERS_IP_PORT_LIST": self.ctx.args.servers,
                "PADDLE_TRAINER_ENDPOINTS": self.ctx.args.trainers,
                "PADDLE_PORT": servers[i].split(":")[1],
                "PADDLE_ROLE": "PSERVER",
                "TRAINING_ROLE": "PSERVER",
                "PADDLE_TRAINERS_NUM": f"{len(trainer_endpoints)}",
                "POD_IP": self.ctx.node.ip,
            }
            e.update(_gloo_envs)
            log_file = f"serverlog.{i}"
            self.add_container(envs=e, log_file=log_file)

        trainer_rank_offset = 0
        for s in trainer_endpoints:
            if s.startswith(host):
                break
            else:
                trainer_rank_offset += 1

        for i in range(trainer_num):
            e = {
                "PADDLE_PSERVERS_IP_PORT_LIST": ",".join(server_endpoints),
                "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints),
                "PADDLE_PORT": trainers[i].split(":")[1],
                "PADDLE_ROLE": "TRAINER",
                "TRAINING_ROLE": "TRAINER",
                "PADDLE_TRAINER_ID": f"{i + trainer_rank_offset}",
                "PADDLE_TRAINERS_NUM": f"{len(trainer_endpoints)}",
                "POD_IP": self.ctx.node.ip,
            }
            e.update(_gloo_envs)
            log_file = f"workerlog.{i}"
            self.add_container(envs=e, log_file=log_file)

    def _build_pod_with_master(self):
        self.pod.rank = int(self.ctx.args.rank)

        server_num = self.ctx.args.server_num or 1
        servers = [
            f"{self.ctx.node.ip}:{p}"
            for p in self.ctx.node.get_free_ports(server_num)
        ]
        trainer_num = self.ctx.args.trainer_num or 1
        trainers = [
            f"{self.ctx.node.ip}:{p}"
            for p in self.ctx.node.get_free_ports(trainer_num)
        ]

        data = json.dumps(
            {
                'name': self.pod.name,
                'rank': self.pod.rank,
                'servers': servers,
                'trainers': trainers,
                'dtype': self.ctx.node.device.dtype,
                'gloo_port': self.ctx.node.get_free_port(),
            }
        )

        peer_list, rank = self.master.sync_peers(
            f'/{self.job.id}/info',
            self.pod.name,
            data,
            self.job.replicas,
            self.pod.rank,
        )

        self.ctx.logger.debug(f"sync peers done {peer_list}")

        peer_list = [json.loads(i) for i in peer_list]

        self.save_pod_log(peer_list)

        server_endpoints = [j for i in peer_list for j in i['servers']]
        trainer_endpoints = [j for i in peer_list for j in i['trainers']]
        # rank_offset = sum([i['replicas'] for i in peer_list[:rank]])

        server_rank_offset = sum([len(i['servers']) for i in peer_list[:rank]])
        trainer_rank_offset = sum(
            [len(i['trainers']) for i in peer_list[:rank]]
        )

        self.pod.rank = rank

        self.pod.replicas = server_num + trainer_num

        import tempfile

        gloo_rendezvous_dir = tempfile.mkdtemp()
        if os.path.exists(gloo_rendezvous_dir):
            shutil.rmtree(gloo_rendezvous_dir)

        gloo_port = peer_list[0]['gloo_port']
        gloo_http = "{}:{}".format(server_endpoints[0].split(":")[0], gloo_port)

        _gloo_envs = {
            "PADDLE_GLOO_RENDEZVOUS": "3",
            "PADDLE_GLOO_FS_PATH": gloo_rendezvous_dir,
            "PADDLE_GLOO_HTTP_ENDPOINT": gloo_http,
            "PADDLE_WITH_GLOO": self.ctx.args.with_gloo,
        }

        for i in range(server_num):
            e = {
                "PADDLE_NNODES": f"{self.job.replicas}",
                "PADDLE_PSERVERS_IP_PORT_LIST": ",".join(server_endpoints),
                "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints),
                "PADDLE_PORT": server_endpoints[i + server_rank_offset].split(
                    ":"
                )[1],
                "PADDLE_ROLE": "PSERVER",
                "TRAINING_ROLE": "PSERVER",
                "PADDLE_TRAINERS_NUM": f"{len(trainer_endpoints)}",
                "POD_IP": self.ctx.node.ip,
            }
            e.update(_gloo_envs)
            log_file = f"serverlog.{i}"
            self.add_container(envs=e, log_file=log_file)

        for i in range(trainer_num):
            e = {
                "PADDLE_NNODES": f"{self.job.replicas}",
                "PADDLE_PSERVERS_IP_PORT_LIST": ",".join(server_endpoints),
                "PADDLE_TRAINER_ENDPOINTS": ",".join(trainer_endpoints),
                "PADDLE_PORT": trainer_endpoints[i + trainer_rank_offset].split(
                    ":"
                )[1],
                "PADDLE_ROLE": "TRAINER",
                "TRAINING_ROLE": "TRAINER",
                "PADDLE_TRAINER_ID": f"{i + trainer_rank_offset}",
                "PADDLE_TRAINERS_NUM": f"{len(trainer_endpoints)}",
                "POD_IP": self.ctx.node.ip,
            }
            e.update(_gloo_envs)
            log_file = f"workerlog.{i}"
            self.add_container(envs=e, log_file=log_file)
        ''' NEW VERSION
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
                "PADDLE_RANK": "{}".format(i + trainer_rank_offset),
            }
            log_tag = "trainer.{}".format(i)
            self.add_container(envs=e, log_tag=log_tag)
        '''
