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

import os
import socket
from collections import namedtuple
import pickle
from contextlib import closing
import time
import datetime

import paddle
import paddle.fluid.core as core
import paddle.distributed as dist
from paddle.distributed.utils.launch_utils import logger
from paddle.distributed.rpc.internal import _serialize, PythonFunc
from paddle.distributed.collective import (
    _new_process_group_impl,
    _default_group_name,
    Group,
)

ServiceInfo = namedtuple("ServiceInfo", ["name", "rank", "ip", "port"])

_DEFAULT_BACKEND = "gloo"
_DEFAULT_TIMEOUT_MS = 500000
_default_group = None
_default_store = None


def _set_default_group(group):
    global _default_group
    _default_group = group


def _set_default_store(store):
    global _default_store
    _default_store = store


def _set_self_info(name, rank, ip, port):
    self_info = pickle.dumps(ServiceInfo(name, rank, ip, port))
    _default_store.set(str(rank), self_info)


def _exchange_all_service_infos():
    world_size = dist.get_world_size()
    all_infos = []
    s = set()
    for rank in range(world_size):
        info = pickle.loads(_default_store.get(str(rank)))
        assert (info.name not in s
                ), "The Worker name must be unique, but name `{}` is repeated."
        s.add(info.name)
        all_infos.append(info)
    return all_infos


def _free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _get_host_ip():
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(socket.getfqdn(hostname))
        return ip
    except:
        return '127.0.0.1'


def _gen_endpoint():
    return "{}:{}".format(_get_host_ip(), _free_port())


def init_rpc(name, rank=None, world_size=None, master_endpoint=None):
    """
    init rpc.

    Arguments:
        name (str): worker name.
        rank (int): worker id.
        world_size (int): number of workers.
        master_endpoint (str): id address of master, other nodes communicate with the master to
            get the information of all service nodes.

    Examples:
        .. code-block:: python
            import paddle.distributed.rpc as rpc

            rpc.init_rpc("worker0", rank=0, world_size=2,
                        master_endpoint="127.0.0.1:8001")
            rpc.shutdown()
    """
    if rank != None:
        rank = rank
        # set environment variable `PADDLE_TRAINER_ID` to reuse dist.get_rank() api
        os.environ["PADDLE_TRAINER_ID"] = str(rank)
    else:
        rank = int(os.environ["PADDLE_TRAINER_ID"])
    if world_size != None:
        # set environment variable `PADDLE_TRAINERS_NUM` to reuse dist.get_world_size() api
        os.environ["PADDLE_TRAINERS_NUM"] = str(world_size)
    else:
        world_size = int(os.environ["PADDLE_TRAINERS_NUM"])
    server_endpoint = os.getenv("PADDLE_SERVER_ENDPOINT", None)
    if server_endpoint is None:
        server_endpoint = _gen_endpoint()
    logger.info("Trainer {}: server endpoint: {}".format(rank, server_endpoint))
    master_endpoint = (master_endpoint if master_endpoint != None else
                       os.environ["PADDLE_MASTER_ENDPOINT"])
    master_addr, master_port = master_endpoint.split(":")
    master_port = int(master_port)
    stop_check_timeout = int(os.getenv("FLAGS_stop_check_timeout", "900"))
    default_store = core.TCPStore(master_addr,
                                  master_port,
                                  rank == 0,
                                  world_size,
                                  timeout=stop_check_timeout)
    _set_default_store(default_store)
    ip, port = server_endpoint.split(":")
    port = int(port)
    _set_self_info(name, rank, ip, port)
    pg = _new_process_group_impl(
        _DEFAULT_BACKEND,
        default_store,
        rank,
        world_size,
        _default_group_name,
        pg_options=None,
    )
    ranks = list(range(world_size))
    group = Group(rank,
                  world_size,
                  id=0,
                  ranks=ranks,
                  pg=pg,
                  name=_default_group_name)
    _set_default_group(group)
    paddle.distributed.barrier(group=group)
    all_infos = _exchange_all_service_infos()
    c_infos = []
    for node_info in all_infos:
        info = core.ServiceInfo(node_info.name, node_info.rank, node_info.ip,
                                node_info.port)
        c_infos.append(info)
    core.init_and_set_agent_instance(name, c_infos)
    core.rpc_start_server()
    paddle.distributed.barrier(group=group)
    core.rpc_start_client()
    logger.info("Trainer {}: Init RPC done!".format(dist.get_rank()))


def rpc_sync(name, fn, timeout_ms=_DEFAULT_TIMEOUT_MS, args=None, kwargs=None):
    """
    Make a blocking RPC call to run function ``fn`` on server ``to``.

    Args:
        to (str): name of the destination server.
        func (fn): a callable function, such as Python callables.
        args (tuple): the argument tuple for the ``fn`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``fn``
                       invocation.
        timeout_ms (float, optional): timeout in milliseconds to use for this RPC.

    Returns:
        Returns the result of running ``fn`` with ``args`` and ``kwargs``.

    Example:
        run on server 0:
            .. code-block:: python

                # On server 0:
                import paddle.distributed.rpc as rpc

                def add(a, b):
                    return a + b

                rpc.init_rpc("worker0", rank=0, world_size=2,
                        master_endpoint="127.0.0.1:8001")
                ret = rpc.rpc_sync("worker1", add, args=(2, 3))
                rpc.shutdown()

        run on server 1:
            .. code-block:: python
                # On server 1:
                import paddle.distributed.rpc as rpc
                rpc.init_rpc("worker1", rank=1, world_size=2,
                        master_endpoint="127.0.0.1:8001")
                rpc.shutdown()
    """
    fut = _invoke_rpc(name, fn, timeout_ms, args, kwargs)
    return fut.wait()


def rpc_async(name, fn, timeout_ms=_DEFAULT_TIMEOUT_MS, args=None, kwargs=None):
    """
    Make a non-blocking RPC call to run function ``fn`` on server ``to``.

    Args:
        to (str): name of the destination server.
        func (fn): a callable function, such as Python callables.
        args (tuple): the argument tuple for the ``fn`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``fn``
                       invocation.
        timeout_ms (float, optional): timeout in milliseconds to use for this RPC.

    Returns:
        Returns a :class:`FutureWrapper` object that can be waited
        on. When completed, the return value of ``fn`` on ``args`` and
        ``kwargs`` can be got by `fut.wait()`

    Example:
        run on server 0:
            .. code-block:: python

                # On server 0:
                import paddle.distributed.rpc as rpc

                def add(a, b):
                    return a + b

                rpc.init_rpc("worker0", rank=0, world_size=2,
                        master_endpoint="127.0.0.1:8001")
                fut = rpc.rpc_async("worker1", add, args=(2, 3))
                print(fut.wait())
                rpc.shutdown()

        run on server 1:
            .. code-block:: python
                # On server 1:
                import paddle.distributed.rpc as rpc
                rpc.init_rpc("worker1", rank=1, world_size=2,
                        master_endpoint="127.0.0.1:8001")
                rpc.shutdown()
    """
    return _invoke_rpc(name, fn, timeout_ms, args, kwargs)


def _invoke_rpc(name, fn, timeout_ms, args, kwargs):
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    serial_obj = _serialize(PythonFunc(fn, args, kwargs))
    future = core.invoke_rpc(name, serial_obj, timeout_ms)
    return future


def _barrier_by_tcp_store():
    import sys
    timeout = sys.maxsize
    global_rank = paddle.distributed.get_rank()
    global_world_size = paddle.distributed.get_world_size()

    if global_world_size < 2:
        return

    barrier_prefix = "Barrier/" + _default_group_name + "/"
    is_master = (global_rank == 0)

    def _check_keys_ready(wait_keys):
        start_time = time.time()
        while len(wait_keys) > 0:
            time.sleep(0.1)
            elapse_time = time.time() - start_time
            if datetime.timedelta(seconds=elapse_time) > timeout:
                raise RuntimeError(
                    "Timeout while initializing process group {}."
                    "Keys {} are not ready sinck rank {} is waiting them."
                    "Two reason may cause this error:\n 1. The create process group api should be called by all ranks.\n"
                    " 2. Try to increase the waiting time.\n".format(
                        _default_group_name, wait_keys, global_rank))
            wait_keys = list(
                filter(lambda key: int(_default_store.get(key)) != 1,
                       wait_keys))

    # all the workers set their exiting key and exit
    # the master will wait for all workers' exiting key, ensure to exit in the end
    if is_master:
        wait_keys = [
            barrier_prefix + str(rank) for rank in range(1, global_world_size)
        ]
        _check_keys_ready(wait_keys)
        _default_store.add(barrier_prefix + str(0), 1)
    else:
        _default_store.add(barrier_prefix + str(global_rank), 1)
        wait_keys = [barrier_prefix + str(0)]
        _check_keys_ready(wait_keys)


def shutdown():
    """
    Perform a shutdown of the RPC agent, stop the server and destroy the agent.
    This will block until all local and remote RPC processes reach this method
    and wait for all outstanding work to complete.

    Examples:
        .. code-block:: python
            import paddle.distributed.rpc as rpc

            rpc.init_rpc("worker0", rank=0, world_size=1,
                        master_endpoint="127.0.0.1:8001")
            rpc.shutdown()
    """
    # never timeout
    _barrier_by_tcp_store()
    core.rpc_stop_server()
    core.rpc_clear_python_rpc_handler()
    logger.info("Trainer {}: rpc shutdown!".format(dist.get_rank()))


def get_service_info(name):
    return core.rpc_get_service_info(name)


def get_all_service_infos():
    return core.rpc_get_all_service_infos()


def get_current_service_info():
    return core.rpc_get_current_service_info()
