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
import six
'''
Master is a distributed store desgin to exchange info among nodes
'''


class Master(object):
    MAIN = "main"
    STANBY = "stanby"
    PATICIPANT = "participant"

    def __init__(self, ctx):
        self.ctx = ctx
        self.server = None
        self.initialized = False
        self.endpoint = None

    def stop(self):
        raise NotImplementedError

    def sync_peers(self, prefix, key, value, size, rank=-1) -> (list, int):
        raise NotImplementedError

    @classmethod
    def factory(cls, ctx):
        if ctx.args.master and ctx.args.master.startswith('etcd://'):
            return ETCDMaster(ctx)
        else:
            return HTTPMaster(ctx)


class HTTPMaster(Master):
    def lazy_init(self):
        if self.initialized:
            return

        self.role = Master.PATICIPANT

        if self.ctx.args.master:
            self.endpoint = self.ctx.args.master
            ip, port = self.endpoint.split(':')
            if ip == self.ctx.node.ip and not self.ctx.node.is_server_ready(
                    ip, int(port)):
                self.server = KVServer(int(port))
                self.role = Master.MAIN
        else:
            port = self.ctx.node.get_free_port()
            self.endpoint = "{}:{}".format(self.ctx.node.ip, port)
            self.server = KVServer(port)
            self.role = Master.MAIN

            print("Copy the following commond to other nodes to run.")
            cmd = [
                sys.executable.split('/')[-1], "-m", "paddle.distributed.run"
            ]
            cmd.extend(["--master", self.endpoint])
            cmd.extend(sys.argv[1:])
            print("-" * 80)
            print(" ".join(cmd))
            print("-" * 80)

            if self.ctx.args.rank >= 0:
                self.ctx.logger.warning(
                    "--rank set in the command may not compatible in auto mode")

        self.client = KVClient(self.endpoint)

        self.initialized = True

        self._start_server()

    def _start_server(self):
        if self.server and not self.server.started:
            self.server.start()
            self.ctx.logger.debug("KV server start at {}".format(self.endpoint))

    def _stop_server(self):
        if self.server and not self.server.stopped:
            self.server.stop()
            self.ctx.logger.debug("KV server stopped")

    def stop(self):
        self._stop_server()

    def sync_peers(self, prefix, key, value, size, rank=-1) -> (list, int):
        if size < 2:
            return [value], 0

        self.lazy_init()

        assert self.client.wait_server_ready(timeout=600), 'server is not ready'

        ky = 'aaaaaa' if rank < 0 and self.role == Master.MAIN else key
        assert self.client.put("{}/{}/{}".format(prefix, ky, rank), value)

        while True:
            rjson = self.client.get_prefix(prefix)
            self.ctx.logger.debug("gather result {}".format(rjson))
            if len(rjson) == size:
                if rank < 0:
                    keys = list(rjson.keys())
                    keys.sort()
                    ret = [rjson[k] for k in keys]
                    idx = ret.index(value)
                    return ret, idx
                else:
                    ret = [None] * size
                    for k, v in rjson.items():
                        ret[int(k.split('/')[-1])] = v
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


class ETCDMaster(Master):
    def __init__(self, ctx):
        super().__init__(ctx)

        if self.ctx.args.master:
            # etcd://localhost:2379
            self.endpoint = self.ctx.args.master.strip("etcd://")

        import etcd3

        host, port = self.endpoint.split(':')
        self.client = etcd3.client(host=host, port=port)

        self.remove_list = set()

    '''
    sync_peers gather all value for key under scop prefix
    result always be sorted either by rank or alphabet of pod.name
    '''

    def sync_peers(self, prefix, key, value, size, rank=-1) -> (list, int):
        self.remove_list.add(prefix)

        path = "{}/{}/{}".format(prefix, key, rank)
        self.client.put(path, six.b(value))
        self.ctx.logger.debug("put path {} value {}".format(path, value))

        while True:
            result = [i for i in self.client.get_prefix(prefix)]
            self.ctx.logger.debug("gather result {}".format(result))

            if len(result) == size:
                if rank < 0:
                    keys = [six.ensure_str(i[1].key) for i in result]
                    sorted_keys = [six.ensure_str(i[1].key) for i in result]
                    sorted_keys.sort()
                    values = [six.ensure_str(i[0]) for i in result]
                    ret = [values[keys.index(k)] for k in sorted_keys]
                    idx = ret.index(value)
                    return ret, idx
                else:
                    ret = [None] * size
                    for v, k in result:
                        ret[int(six.ensure_str(k.key).split('/')[
                            -1])] = six.ensure_str(v)
                    return ret, rank
            else:
                time.sleep(0.5)

    def broadcast(self, key, value=None):
        pass

    def stop(self):
        ## TODO(MAYBE LATE)
        for i in self.remove_list:
            self.client.delete_prefix(i)
