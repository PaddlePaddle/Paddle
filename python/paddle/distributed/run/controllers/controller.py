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

import sys

from paddle.distributed.run.job import Job
from paddle.distributed.run.job import Pod
from paddle.distributed.run.job import Container

from paddle.distributed.run.utils.kv_client import KVClient
from paddle.distributed.run.utils.kv_server import KVServer

import time


class Controller(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self._kv_server = None

        self.pod = Pod()
        self.job = Job()

        self.ctx.logger.debug(self.pod)

        self.build_job()
        self.build_pod()

    def _run_kv_server(self, port=None):
        if self._kv_server:
            return

        port = port if port else self.ctx.node.get_free_port()
        self._kv_server = KVServer(port)
        self._kv_server.start()

        host = self.ctx.node.ip
        self.job.master = "{}:{}".format(host, port)
        self.ctx.logger.debug("KV server start at {}".format(port))

    def _stop_kv_server(self):
        if not self._kv_server:
            return

        self._kv_server.stop()
        self.ctx.logger.debug("KV server stopped")

    def sync_pods(self):

        if self.pod.rank == 0:
            self._run_kv_server()

        cli = KVClient(self.job.master)

        host = self.ctx.node.ip
        replicas = self.ctx.node.device.count  # self.pod.replicas
        eps = [
            "{}:{}".format(host, p)
            for p in self.ctx.node.get_free_ports(replicas)
        ]

        self.ctx.logger.debug("eps {}".format(eps))

        prefix = "/workers/"
        assert cli.put("{}{}".format(prefix, self.pod.name), ",".join(eps))

        while True:
            ret = cli.get_prefix(prefix)
            print("ret", ret)
            if len(ret) == self.job.replicas:
                break
            else:
                time.sleep(1)

        if self.pod.rank == 0:
            self._stop_kv_server()

    def build_job(self):
        pass

    def build_pod(self):
        raise NotImplementedError

    def _get_entrypoint(self):
        entrypoint = [sys.executable, "-u", self.ctx.args.training_script]
        entrypoint.extend(self.ctx.args.training_script_args)
        return entrypoint

    def _build_container(self, entrypoint=None, envs={}, use_ctx_env=True):
        c = Container()
        c.entrypoint = entrypoint or self._get_entrypoint()
        c.env = self.ctx.envs
        c.update_env(envs)

        return c

    def run(self):
        self.ctx.logger.info("env {}".format(self.pod.containers[0].env))
        self.pod.create()
