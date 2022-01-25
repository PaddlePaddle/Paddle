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


class PSController(Controller):
    @classmethod
    def enable(cls, ctx):
        ctx.logger.debug("PSController enabled")
        if ctx:
            return True
        else:
            return False

    def build_job(self):

        self.job.replicas = self.ctx.args.np or 1

        self.job.endpoints = []

        if self.ctx.args.ips:
            self.job.ips = self.ctx.args.ips.split(',')
            self.job.replicas = self.ctx.args.np or len(self.job.ips)

    def build_pod(self):

        host = self.ctx.node.ip
        self.pod.replicas = self.ctx.node.device.count

        ips = self.ctx.args.ips.split(',')
        assert ips.index(host) >= 0
        if len(ips) > 1:
            start_port = 6170
            self.job.endpoints = [
                "{}:{}".format(i, 6170 + p)
                for p in range(self.pod.replicas) for i in ips
            ]

        self.pod.endpoints = [
            "{}:{}".format(host, 6170 + p) for p in range(self.pod.replicas)
        ]
        '''
        self.pod.endpoints = [
            "{}:{}".format(self.ctx.node.ip, p)
            for p in self.ctx.node.get_free_ports(self.pod.replicas)
        ]
        '''

        self.ctx.logger.debug(self.pod)
        '''
        if self.enable_join():
            self.ctx.logger.debug("join mode enabled")
            self.sync_pods()
        '''

        if not self.job.endpoints:
            self.job.endpoints = self.pod.endpoints.copy()

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
