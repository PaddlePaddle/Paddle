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

from paddle.distributed.run.utils.kv_client import KVClient
from paddle.distributed.run.utils.kv_server import KVServer

import time
import sys
'''
Store is a distributed store desgin to exchange info among nodes
'''


class Store(object):
    MAIN = "main"
    STANBY = "stanby"
    PATICIPANT = "participant"

    def __init__(self, ctx):
        self.ctx = ctx
        self.server = None
        self.initialized = False
        self.master = None

    def lazy_init(self):
        if self.initialized:
            return

        self.role = Store.PATICIPANT

        if self.ctx.args.master:
            self.master = self.ctx.args.master
            ip, port = self.master.split(':')
            if ip == self.ctx.node.ip and not self.ctx.node.is_server_ready(
                    ip, int(port)):
                self.server = KVServer(int(port))
                self.role = Store.MAIN
        else:
            port = self.ctx.node.get_free_port()
            self.master = "{}:{}".format(self.ctx.node.ip, port)
            self.server = KVServer(port)
            self.role = Store.MAIN

            print("Copy the following commond to other nodes to run.")
            cmd = [
                sys.executable.split('/')[-1], "-m", "paddle.distributed.run"
            ]
            cmd.extend(["--master", self.master])
            cmd.extend(sys.argv[1:])
            print("-" * 80)
            print(" ".join(cmd))
            print("-" * 80)

            if self.ctx.args.rank >= 0:
                self.ctx.logger.warning(
                    "--rank set in the command may not compatible in auto mode")

        self.client = KVClient(self.master)

        self.initialized = True

        self.start_server()

    def start_server(self):
        if self.server and not self.server.started:
            self.server.start()
            self.ctx.logger.debug("KV server start at {}".format(self.master))

    def stop_server(self):
        if self.server and not self.server.stopped:
            self.server.stop()
            self.ctx.logger.debug("KV server stopped")

    '''
    allgather gather all value for key under scop prefix
    if rank is set, result will be sort
    '''

    def allgather(self, prefix, key, value, size, rank=-1) -> (list, int):
        assert '-' not in prefix, 'prefix cannot contains -'
        assert '-' not in key, 'key cannot contains -'

        if size < 2:
            return value.split(",")

        self.lazy_init()

        assert self.client.wait_server_ready(timeout=600), 'server is not ready'
        assert self.client.put("{}{}-{}".format(prefix, key, rank), value)

        while True:
            rjson = self.client.get_prefix(prefix)
            if len(rjson) == size:
                if rank < 0:
                    ret = list(rjson.values())
                    ret.sort()
                    idx = ret.index(value)
                    ret[0], ret[idx] = ret[idx], ret[0]
                    return ret, idx
                ret = [None] * size
                for k, v in rjson.items():
                    ret[int(k.split('-')[-1])] = v
                return ret, rank
            else:
                time.sleep(0.5)

    def broadcast(self, key, value=None):
        if value:
            self.client.put(key, value)

        while True:
            r = self.client.get(key)
            if r != "":
                return r
            else:
                time.sleep(0.5)


class ETCDStore(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.server = None
        self.initialized = False
        self.master = None

        import etcd3

    '''
    allgather gather all value for key under scop prefix
    if rank is set, result will be sort
    '''

    def allgather(self, prefix, key, value, size, rank=-1) -> (list, int):
        pass

    def broadcast(self, key, value=None):
        pass
