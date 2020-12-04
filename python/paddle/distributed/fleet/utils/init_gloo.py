#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Defination of gloo initialization with http server."""
import time
from multiprocessing import Process
import paddle.fluid as fluid


def init_gloo_with_http(ip, port, prefix, iface, init_timeout_sec,
                        run_timeout_sec, worker_num, server_num, rank, role,
                        start_http_server, need_init_all, http_server_d):
    """
    Initialize gloo using http server.
    Args:
        ip (str): ip address of the trainer with rank id of 0;
        port (int): ip port of the trainer with rank id of 0;
        prefix (str): prefix of the url path;
        iface (str): network card used for gloo initialization;
        init_timeout_sec (int): timeout during initialization;
        run_timeout_sec (int): timeout during running;
        worker_num (int): number of trainers with the role of worker;
        server_num (int): number of trainers with the role of server;
        rank (int): rank id of the current trainer;
        rank_num (int): number of ranks;
        role (str): role of the trainer, one of "WORKER" or "SERVER";
        start_http_server (bool): whether to start http server;
        need_init_all (bool): whether to initalize all workers and servers;
        http_server_d (dict): a dict to record the status of the http server;
    """

    def __start_kv_server(http_server_d, size_d):
        from paddle.distributed.fleet.utils.http_server import KVServer
        http_server = KVServer(port, size_d)
        http_server.start()
        wait_seconds = 3
        while http_server_d.get("running",
                                False) or not http_server.should_stop():
            time.sleep(wait_seconds)
        http_server.stop()

    def init_kv_server(http_server_d):
        size_d = {
            "trainer": worker_num,
            "pserver": ps_num,
            "all": worker_num + ps_num
        }

        http_server_d["running"] = True
        # child process for http server
        _http_server = Process(
            target=__start_kv_server, args=(http_server_d, size_d))
        _http_server.daemon = True
        # start child process
        _http_server.start()
        return _http_server

    def init(rank, nodes, role, ip, port):
        gloo_strategy = fluid.core.GlooParallelStrategy()
        gloo_strategy.rank = rank
        gloo_strategy.rank_num = nodes
        gloo_strategy.ip_address = ip
        gloo_strategy.ip_port = port
        gloo_strategy.init_seconds = init_timeout_seconds
        gloo_strategy.run_seconds = run_timeout_seconds
        gloo_strategy.scope = role
        gloo = fluid.core.GlooParallelContext(gloo_strategy)
        gloo.init()
        return gloo

    port = int(port)

    if start_http_server:
        http_server = init_kv_server(http_server_d)

    worker_comm = None
    server_comm = None
    nodes_comm = None
    if role == "WORKER":
        gloo = init(rank, worker_num, "WORKER")
        worker_comm = gloo
    else:
        gloo = init(rank, server_num, "SERVER")
        server_comm = gloo

    if need_init_all:
        rank = rank + worker_num
        rank_num = worker_num + server_num
        gloo = init(rank, rank_num, "ALL")
        nodes_comm = gloo

    if start_http_server:
        http_server_d["running"] = False
        http_server.join()
    return http_server, worker_comm, server_comm, nodes_comm
