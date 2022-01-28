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
        if ctx.args.server_num or len(ctx.args.servers) > 0:
            return True
        else:
            return False

    def build_job(self):

        self.job.replicas = self.ctx.args.np or 1

        self.job.endpoints = []

    def build_pod(self):

        host = self.ctx.node.ip
        #self.pod.replicas = self.ctx.node.device.count

        server_num = ctx.args.server_num or len(ctx.args.servers)

        for i in range(server_num):
            e = {
                "PADDLE_PSERVERS": server_endpoints,
                "PADDLE_TRAINER_ENDPOINTS": worker_endpoints,
                "PADDLE_PORT": cur_server.endpoint.split(":")[1],
                "TRAINING_ROLE": "PSERVER",
                "PADDLE_TRAINERS_NUM": str(self.worker_num),
                "POD_IP": host,
                "PADDLE_WITH_GLOO": str(os.getenv("PADDLE_WITH_GLOO", "0")),
                "PADDLE_GLOO_RENDEZVOUS": "3",
                "PADDLE_GLOO_FS_PATH": self.gloo_rendezvous_dir,
                "PADDLE_GLOO_HTTP_ENDPOINT": self.http_port
            }
            self.add_container(envs=e)
