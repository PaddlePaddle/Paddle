# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import warnings
from multiprocessing import Process, Manager

# deprecated module import
from paddle.fluid import core
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready

__all__ = []

_global_gloo_ctx = None


def _start_kv_server(port, http_server_d, size):
    from paddle.distributed.fleet.utils.http_server import KVServer
    http_server = KVServer(int(port), size=size)
    http_server.start()
    wait_seconds = 3
    while http_server_d.get("running", False) or not http_server.should_stop():
        time.sleep(wait_seconds)
    http_server.stop()


def init_gloo_parallel_env(rank_id, rank_num, server_endpoint, with_gloo=True):
    """
    Initialize parallel environment with gloo.

    Args:
        rank_id: int, the index of current rank
        rank_num: int, the number of ranks in this parallel env
        server_endpoint: str, endpoint of server to init gloo context in ip:port format
        with_gloo: bool, True as default

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle

            # initialize a parallel environment for a job using 2 ranks
            rank_num = 2
            server_endpoint = "127.0.0.1:8080"

            # process 1
            rank_id = 0
            paddle.distributed.init_gloo_parallel_env(
                rand_id, rank_num, server_endpoint)

            # process 2
            rank_id = 1
            paddle.distributed.init_gloo_parallel_env(
                rand_id, rank_num, server_endpoint)
    """

    assert (rank_num < 2) is False, \
        "rank_num should greater than or equal to 2 for parallel environment initialzation."
    assert with_gloo is True, "flag with_gloo is not set!"

    # init gloo context
    manager = Manager()
    # global dict to store status
    http_server_status = manager.dict()
    http_server_status["running"] = False
    if rank_id == 0:
        # The scope for worker used by http server is '_worker'
        size = {'_worker': rank_num}
        http_server_proc = Process(
            target=_start_kv_server,
            args=(int(server_endpoint.split(":")[1]), http_server_status, size))
        http_server_proc.daemon = True
        http_server_status["running"] = True
        http_server_proc.start()

    # all processes in this parallel environment should wait until server is ready
    wait_server_ready([server_endpoint])

    gloo_strategy = core.GlooParallelStrategy()
    gloo_strategy.rank = rank_id
    gloo_strategy.rank_num = rank_num
    gloo_strategy.ip_address = server_endpoint.split(":")[0]
    gloo_strategy.ip_port = int(server_endpoint.split(":")[1])
    # default_init_timeout_seconds
    gloo_strategy.init_seconds = 3600
    # default_run_timeout_seconds
    gloo_strategy.run_seconds = 9999999

    global _global_gloo_ctx
    _global_gloo_ctx = core.GlooParallelContext(gloo_strategy)
    _global_gloo_ctx.init()

    if rank_id == 0:
        http_server_status["running"] = False
        http_server_proc.join()


def barrier_func():
    """
    Call barrier function with initialized gloo context.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed import init_gloo_parallel_env

            # initialize a parallel environment for a job using 2 ranks
            rank_num = 2
            server_endpoint = "127.0.0.1:8080"

            # process 1
            rank_id = 0
            paddle.distributed.init_gloo_parallel_env(
                rand_id, rank_num, server_endpoint)
            paddle.distributed.barrier_func()

            # process 2
            rank_id = 1
            paddle.distributed.init_gloo_parallel_env(
                rand_id, rank_num, server_endpoint)
            paddle.distributed.barrier_func()
    """

    _global_gloo_ctx.barrier()


def release_gloo(rank_id):
    """
    Release the parallel environment initialized by gloo

    Args:
        rank_id: int, only the rank with id == 0 can do the release.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed import init_gloo_parallel_env

            # initialize a parallel environment for a job using 2 ranks
            rank_num = 2
            server_endpoint = "127.0.0.1:8080"

            # process 1
            rank_id = 0
            paddle.distributed.init_gloo_parallel_env(
                rand_id, rank_num, server_endpoint)
            paddle.distributed.barrier_func()
            paddle.distributed.release_gloo(rank_id)

            # process 2
            rank_id = 1
            paddle.distributed.init_gloo_parallel_env(
                rand_id, rank_num, server_endpoint)
            paddle.distributed.barrier_func()
            paddle.distributed.release_gloo(rank_id)
    """

    if (rank_id == 0):
        _global_gloo_ctx.release()
