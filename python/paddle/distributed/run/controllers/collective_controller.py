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
from paddle.distributed.run.base import Job
from paddle.distributed.run.base import Pod
from paddle.distributed.run.base import Container


class CollectiveController(Controller):
    @classmethod
    def enable(cls, ctx):
        ctx.logger.debug("CollectiveController enabled")
        if ctx:
            return True
        else:
            return False

    def __init__(self, ctx):
        ctx.logger.info("CollectiveController init")
        self.ctx = ctx
        self.pod = Pod()
        self.job = Job()

        self._build_job()
        self._build_pod()

    def _build_job(self):
        # local node only
        self.job.endpoints = []
        #self.ips = [self.ctx.node.ip for _ in range(self.ctx.node.device.count)]

    def _build_pod(self):
        self.pod.replicas = self.ctx.node.device.count
        entrypoint = [self.ctx.args.training_script]
        entrypoint.extend(self.ctx.args.training_script_args)

        for i in range(self.pod.replicas):
            c = Container()
            c.entrypoint = entrypoint
            c.env = self.ctx.envs
            c.update_env({
                "PADDLE_TRAINER_ENDPOINTS": ",".join(self.job.endpoints),
                "PADDLE_CURRENT_ENDPOINT": "%s" % self.ctx.node.ip,
                "PADDLE_TRAINER_ID":
                "%d" % (self.pod.rank * self.pod.replicas + i),
                "PADDLE_TRAINERS_NUM": "%d" % self.pod.replicas,
                "PADDLE_RANK_IN_NODE": str(i),
            })

            self.pod.add_container(c)

    def run(self):
        self.ctx.logger.info("env {}".format(self.pod.containers[0].env))
        self.pod.create()
