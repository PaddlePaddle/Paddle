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

from paddle.distributed.launch.utils.kv_client import KVClient
from paddle.distributed.launch.utils.kv_server import KVServer

import time
import sys
import six
import threading
import copy
import random

ETCD_PROTOCAL = 'etcd://'


class Master(object):
    '''
    Master is a distributed store design to exchange info among nodes
    '''

    MAIN = "main"
    STANDBY = "standby"
    PATICIPANT = "participant"

    def __init__(self, ctx):
        self.ctx = ctx
        self.server = None
        self.initialized = False
        self.endpoint = None

    def stop(self):
        raise NotImplementedError

    def set_status(self, status):
        pass

    def get_status(self):
        return None

    def restart_peer(self):
        pass

    def sync_peers(self, prefix, key, value, size, rank=-1) -> (list, int):
        raise NotImplementedError

    @classmethod
    def factory(cls, ctx):
        if ctx.args.master and ctx.args.master.startswith(ETCD_PROTOCAL):
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
            if ip in ['127.0.0.1', self.ctx.node.ip]:
                time.sleep(2 * random.random())
                while not self.ctx.node.is_server_ready(ip, int(port)):
                    try:
                        self.server = KVServer(int(port))
                        self.role = Master.MAIN
                        break
                    except Exception as e:
                        self.ctx.logger.warning("start master failed {}".format(
                            e))
                        time.sleep(0.1)
                        continue
        else:
            port = self.ctx.node.get_free_port()
            self.endpoint = "{}:{}".format(self.ctx.node.ip, port)
            self.server = KVServer(port)
            self.role = Master.MAIN

            print("Copy the following command to other nodes to run.")
            cmd = [
                sys.executable.split('/')[-1], "-m", "paddle.distributed.launch"
            ]
            cmd.extend(["--master", self.endpoint])
            cmd.extend(sys.argv[1:])
            print("-" * 80)
            print(" ".join(cmd))
            print("-" * 80)

            if self.ctx.args.rank >= 0:
                self.ctx.logger.warning(
                    "--rank set in the command may not compatible in auto mode")

        if '127.0.0.1' in self.endpoint:
            self.endpoint = self.endpoint.replace('127.0.0.1', self.ctx.node.ip)
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

        self.ctx.logger.info("Waiting peer start...")

        self.lazy_init()

        while not self.ctx.status.is_done():
            if self.client.wait_server_ready(timeout=5):
                break
            else:
                self.ctx.logger.warning("master not ready")
                time.sleep(0.1)

        # 'aaaaaa' make sure main pod (master server) as rank 0
        ky = 'aaaaaa' if rank < 0 and self.role == Master.MAIN else key
        k = "{}/{}/{}".format(prefix, ky, rank)

        while not self.ctx.status.is_done():
            if not self.client.put(k, value):
                self.ctx.logger.warning("put value failed")
                time.sleep(0.1)
                continue

            rjson = self.client.get_prefix(prefix)
            self.ctx.logger.debug("sync peers {}".format(rjson))
            if rjson and len(rjson) == size:
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
        return [], 0


class ETCDMaster(Master):
    def __init__(self, ctx):
        super().__init__(ctx)

        if self.ctx.args.master:
            # etcd://localhost:2379
            self.endpoint = self.ctx.args.master.strip("etcd://")

        import etcd3

        host, port = self.endpoint.split(':')
        self.client = etcd3.client(host=host, port=port)

    def sync_peers(self, prefix, key, value, size, rank=-1) -> (list, int):
        '''
        sync_peers gather all value for key under scope prefix
        result always be sorted either by rank or alphabet of pod.name
        '''

        if size < 2:
            return [value], 0

        self.ctx.logger.info("Waiting peer start...")

        path = "{}/{}/{}".format(prefix, key, rank)

        self.client.delete_prefix(prefix)

        self.ctx.logger.debug("sync path {} value {}".format(path, value))

        while not self.ctx.status.is_done():
            self.client.put(path, six.b(value))

            result = [i for i in self.client.get_prefix(prefix)]
            result = copy.deepcopy(result)
            self.ctx.logger.debug("sync peers {}".format(result))

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
                        ii = int(six.ensure_str(k.key).split('/')[-1])
                        if ii < 0:
                            self.ctx.logger.error(
                                "rank {} error in sync".format(ii))
                        ret[ii] = six.ensure_str(v)
                    return ret, rank
            else:
                time.sleep(0.5)

    def register_heartbeat(self, job_id, pod_id, ttl=10):
        if hasattr(self, 'heartbeat_prefix'):
            self.ctx.logger.warning("Heartbeat already done")
            return

        self.job_prefix = '/paddle/{}'.format(job_id)
        self.heartbeat_prefix = '{}/heartbeat'.format(self.job_prefix)

        lease = self.client.lease(ttl)

        #self.client.delete_prefix(self.job_prefix)

        beat_path = "{}/{}".format(self.heartbeat_prefix, pod_id)
        self.client.put(beat_path, six.b(pod_id), lease=lease)

        def _beat_watch(event):
            self.ctx.status.restart()

        beat_watch = self.client.add_watch_prefix_callback(
            self.heartbeat_prefix, _beat_watch)

        def _heartbeat():
            while not self.ctx.status.is_done():
                try:
                    lease.refresh()
                    if pod_id not in self.fetch_peer_alive():
                        self.client.put(beat_path, six.b(pod_id), lease=lease)
                        self.ctx.logger.debug("Heartbeat register again")
                except Exception as e:
                    self.ctx.logger.error("Heartbeat error {}".format(e))
                time.sleep(ttl / 2)
            self.ctx.logger.debug("Heartbeat done")
            self.client.cancel_watch(beat_watch)

        self.beat_thread = threading.Thread(
            name='heartbeat', target=_heartbeat, daemon=True)
        self.beat_thread.start()

    def fetch_peer_alive(self):
        peer_alive = [
            six.ensure_str(i[0])
            for i in self.client.get_prefix(self.heartbeat_prefix)
        ]
        self.ctx.logger.debug("peer alive {}".format(peer_alive))
        return peer_alive

    def wait_peer_ready(self, replicas_min, replicas_max, timeout):
        end = time.time() + timeout
        while not self.ctx.status.is_done() and time.time() < end:
            if len(self.fetch_peer_alive()) == replicas_max:
                return (True, replicas_max)
            else:
                time.sleep(0.5)

        np = len(self.fetch_peer_alive())
        if np >= replicas_min and np <= replicas_max:
            return (True, np)
        else:
            return (False, np)

    def restart_peer(self):
        self.client.delete_prefix(self.heartbeat_prefix)

    def set_status(self, status):
        assert self.client.put(
            self.job_prefix, six.b(status),
            lease=self.client.lease(600)), "set status failed {}".format(status)

    def get_status(self):
        return six.ensure_str(self.client.get(self.job_prefix)[0] or '')

    def stop(self):
        if hasattr(self, 'beat_thread'):
            self.ctx.status.done()
            # TODO(kuizhiqing) thread should exit
            #self.beat_thread.join()
