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
from collections import namedtuple
import pickle
import time
import datetime

import paddle.fluid.core as core
from paddle.distributed.utils.launch_utils import logger
from paddle.distributed.rpc.internal import _serialize, PythonFunc
from paddle.distributed.launch.context import Node

ServiceInfo = namedtuple("ServiceInfo", ["name", "rank", "ip", "port"])

_DEFAULT_TIMEOUT_MS = 500000
_TIMEOUT_MAX_DAYS = 99999999
_default_store = None
# count the number of `_barrier_nver_timeout` is called and
# ensure that the barrier key is unique
_barrier_count = 0


def _set_default_store(store):
    global _default_store
    _default_store = store


def _set_self_info(name, rank, ip, port):
    self_info = pickle.dumps(ServiceInfo(name, rank, ip, port))
    _default_store.set(str(rank), self_info)


def _exchange_all_service_infos(world_size):
    all_infos = []
    s = set()
    for rank in range(world_size):
        info = pickle.loads(_default_store.get(str(rank)))
        assert (info.name not in s
                ), "The Worker name must be unique, but name `{}` is repeated."
        s.add(info.name)
        all_infos.append(info)
    return all_infos


def _gen_endpoint():
    node = Node()
    ip = node.get_host_ip()
    free_port = node.get_free_port()
    return "{}:{}".format(ip, free_port)


def init_rpc(name, rank=None, world_size=None, master_endpoint=None):
    """
    init rpc.

    Args:
        name (str): worker name.
        rank (int): worker id.
        world_size (int): number of workers.
        master_endpoint (str): id address of master, other nodes communicate with the master to
            get the information of all service nodes.

    Returns:
        None.

    Examples:
        .. code-block:: python
            import paddle.distributed.rpc as rpc

            rpc.init_rpc("worker0", rank=0, world_size=2,
                        master_endpoint="127.0.0.1:8001")
            rpc.shutdown()
    """
    if rank == None:
        rank = int(os.environ["PADDLE_TRAINER_ID"])
    if world_size == None:
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
    all_infos = _exchange_all_service_infos(world_size)
    c_infos = []
    for node_info in all_infos:
        info = core.ServiceInfo(node_info.name, node_info.rank, node_info.ip,
                                node_info.port)
        c_infos.append(info)
    core.init_and_set_agent_instance(name, c_infos)
    core.rpc_start_server()
    _barrier_never_timeout(rank, world_size)
    core.rpc_start_client()
    logger.info("Trainer {}: Init RPC done!".format(rank))


def rpc_sync(to, fn, args=None, kwargs=None, timeout_ms=_DEFAULT_TIMEOUT_MS):
    """
    Make a blocking RPC call to run function ``fn`` on server ``to``.

    Args:
        to (str): name of the destination server.
        fn (fn): a callable function, such as Python callables.
        args (tuple): the argument tuple for the ``fn`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``fn``
                       invocation.
        timeout_ms (float, optional): timeout in milliseconds to use for this RPC.

    Returns:
        Returns the result of running ``fn`` with ``args`` and ``kwargs``.

    Examples:
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
    fut = _invoke_rpc(to, fn, args, kwargs, timeout_ms)
    return fut.wait()


def rpc_async(to, fn, args=None, kwargs=None, timeout_ms=_DEFAULT_TIMEOUT_MS):
    """
    Make a non-blocking RPC call to run function ``fn`` on server ``to``.

    Args:
        to (str): name of the destination server.
        fn (fn): a callable function, such as Python callables.
        args (tuple): the argument tuple for the ``fn`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``fn``
                       invocation.
        timeout_ms (float, optional): timeout in milliseconds to use for this RPC.

    Returns:
        Returns a :class:`FutureWrapper` object that can be waited
        on. When completed, the return value of ``fn`` on ``args`` and
        ``kwargs`` can be got by `fut.wait()`.

    Examples:
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
    return _invoke_rpc(to, fn, args, kwargs, timeout_ms)


def _invoke_rpc(to, fn, args, kwargs, timeout_ms):
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    serial_obj = _serialize(PythonFunc(fn, args, kwargs))
    future = core.invoke_rpc(to, serial_obj, timeout_ms)
    return future


def _barrier_never_timeout(global_rank, global_world_size):
    # max timeout
    timeout = datetime.timedelta(days=_TIMEOUT_MAX_DAYS)

    if global_world_size < 2:
        return

    global _barrier_count
    barrier_prefix = "Barrier/" + str(_barrier_count) + "/"
    _barrier_count += 1
    is_master = (global_rank == 0)

    def _check_keys_ready(wait_keys):
        start_time = time.time()
        while len(wait_keys) > 0:
            time.sleep(0.1)
            elapse_time = time.time() - start_time
            if datetime.timedelta(seconds=elapse_time) > timeout:
                raise RuntimeError(
                    "Keys {} are not ready sinck rank {} is waiting them.".
                    format(wait_keys, global_rank))
            wait_keys = list(
                filter(lambda key: int(_default_store.get(key)) != 1,
                       wait_keys))

    if is_master:
        # the master will add key, wait for all workers'exiting key and exit in the end.
        # Note: the master must exit in the end to ensure that the TcpServer is destroyed in the end.
        wait_keys = [
            barrier_prefix + str(rank) for rank in range(1, global_world_size)
        ]
        _default_store.add(barrier_prefix + str(0), 1)
        _check_keys_ready(wait_keys)
    else:
        wait_keys = [barrier_prefix + str(0)]
        _check_keys_ready(wait_keys)
        _default_store.add(barrier_prefix + str(global_rank), 1)


def shutdown():
    """
    Perform a shutdown of the RPC agent, stop the server and destroy the agent.
    This will block until all local and remote RPC processes reach this method
    and wait for all outstanding work to complete.

    Returns:
        None.

    Examples:
        .. code-block:: python
            import paddle.distributed.rpc as rpc

            rpc.init_rpc("worker0", rank=0, world_size=1,
                        master_endpoint="127.0.0.1:8001")
            rpc.shutdown()
    """
    info = get_current_service_info()
    rank = info.rank
    world_size = len(get_all_service_infos())
    # master will exit in the end
    _barrier_never_timeout(rank, world_size)
    core.rpc_stop_server()
    logger.info("Trainer {}: rpc shutdown!".format(rank))


def get_service_info(name):
    """
    Get service information by service name.

    Args:
        name (str): name of the server.

    Returns:
        class `ServiceInfo` with attribute `name`, `rank`, `ip` and `port`.

    Examples:
        run on server `11.11.11.10`
        .. code-block:: python
            import paddle.distributed.rpc as rpc
            import os

            os.environ["PADDLE_SERVER_ENDPOINT"] = "11.11.11.10:8002"
            rpc.init_rpc("worker0", rank=0, world_size=1,
                        master_endpoint="127.0.0.1:8001")

            print(rpc.get_service_info("worker0))
            # {name: worker0, rank: 0, ip: 11.11.11.10, port: 8002}

            rpc.shutdown()
    """
    return core.rpc_get_service_info(name)


def get_all_service_infos():
    """
    Get all service informations.

    Returns:
        List[ServiceInfo].

    Examples:
        run on server `11.11.11.10`:
            .. code-block:: python

                # On server 0:
                import paddle.distributed.rpc as rpc
                import os

                os.environ["PADDLE_SERVER_ENDPOINT"] = "11.11.11.10:8002"
                rpc.init_rpc("worker0", rank=0, world_size=2,
                        master_endpoint="11.11.11.10:8001")

                print(rpc.get_all_service_infos())
                # [{name: worker0, rank: 0, ip: 11.11.11.10, port: 8002},
                # {name: worker1, rank: 1, ip: 11.11.11.11, port: 8002}]

                rpc.shutdown()

        run on server `11.11.11.11`:
            .. code-block:: python
                # On server 1:
                import paddle.distributed.rpc as rpc
                import os

                os.environ["PADDLE_SERVER_ENDPOINT"] = "11.11.11.11:8002"
                rpc.init_rpc("worker1", rank=1, world_size=2,
                        master_endpoint="11.11.11.10:8001")
                rpc.shutdown()
    """
    return core.rpc_get_all_service_infos()


def get_current_service_info():
    """
    Get current service information.

    Returns:
        class `ServiceInfo` with attribute `name`, `rank`, `ip` and `port`.

    Examples:
        run on server `11.11.11.10`
        .. code-block:: python
            import paddle.distributed.rpc as rpc
            import os

            os.environ["PADDLE_SERVER_ENDPOINT"] = "11.11.11.10:8002"
            rpc.init_rpc("worker0", rank=0, world_size=1,
                        master_endpoint="127.0.0.1:8001")

            print(rpc.get_current_service_info())
            # {name: worker0, rank: 0, ip: 11.11.11.10, port: 8002}

            rpc.shutdown()
    """
    return core.rpc_get_current_service_info()
