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

from .controller import ControleMode, Controller


class RpcController(Controller):
    @classmethod
    def enable(cls, ctx):
        if ctx.args.run_mode == ControleMode.RPC:
            ctx.logger.debug(f"{cls.__name__} enabled")
            return True
        else:
            return False

    def build_pod(self):
        assert (
            self.ctx.args.master is not None
        ), "Master is None, Please set master address!"
        self._build_pod_with_master()

    def _build_pod_with_master(self):
        # nproc_per_node
        self.pod.replicas = self.pod_replicas()

        # rank will be reset when restart
        self.pod.rank = int(self.ctx.args.rank)

        port = self.ctx.node.get_free_port()

        # compatible
        endpoints = [
            f"{self.ctx.node.ip}:{p}"
            for p in self.ctx.node.get_free_ports(self.pod.replicas)
        ]

        data = json.dumps(
            {
                "name": self.pod.name,
                "rank": self.pod.rank,
                "replicas": self.pod.replicas,
                "dtype": self.ctx.node.device.dtype,
                "candidate": f"{self.ctx.node.ip}:{port}",
                "endpoints": ",".join(endpoints),
            }
        )
        peer_list, rank = self.master.sync_peers(
            f"/{self.job.id}/info",
            self.pod.name,
            data,
            self.job.replicas,
            self.pod.rank,
        )
        self.pod.rank = rank

        if len(peer_list) < 1:
            return False

        peer_list = [json.loads(i) for i in peer_list]

        self.ctx.logger.debug(f"sync peers done {peer_list}")
        self.save_pod_log(peer_list)

        global_size = sum([i["replicas"] for i in peer_list])
        rank_offset = sum([i["replicas"] for i in peer_list[:rank]])

        rpc_master = peer_list[0]["candidate"]
        self.pod.reset()
        for i in range(self.pod.replicas):
            e = {
                "PADDLE_MASTER_ENDPOINT": rpc_master,
                "PADDLE_WORKER_ENDPOINT": endpoints[i],
                "PADDLE_TRAINER_ID": f"{i + rank_offset}",
                "PADDLE_TRAINERS_NUM": f"{global_size}",
            }
            log_file = f"workerlog.{i + rank_offset}"
            self.add_container(envs=e, log_file=log_file)
        return True
