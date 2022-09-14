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

import paddle.fluid.core as core
from paddle.distributed.fleet.rpc.internal import serialize, deserialize, PythonFunc
from paddle.distributed.fleet.rpc.utils import barrier, exchange_service_info

MASTER_ADDR = None
MASTER_PORT = None


def init_rpc(name,
             rank=None,
             world_size=None,
             server_endpoint=None,
             master_endpoint=None):
    """
    init rpc.

    Arguments:
        name: worker name
        current_endpoint (str): ip address of server(ip:port)
        master_endport (str): id address of master, other nodes communicate with the master to
            get the information of all service nodes.
    """
    rank = rank if rank != None else int(os.environ["PADDLE_RANK"])
    world_size = (world_size if world_size != None else int(
        os.environ["PADDLE_WORLD_SIZE"]))
    server_endpoint = (server_endpoint if server_endpoint != None else
                       os.environ["PADDLE_SERVER_ENDPOINT"])
    master_endpoint = (master_endpoint if master_endpoint != None else
                       os.environ["PADDLE_MASTER_ENDPOINT"])
    master_addr, master_port = master_endpoint.split(":")
    master_port = int(master_port)
    global MASTER_ADDR, MASTER_PORT
    MASTER_ADDR, MASTER_PORT = master_addr, master_port
    if rank == 0:
        assert (server_endpoint != master_endpoint
                ), "current endpoint: {} must != master endpoint: {}".format(
                    server_endpoint, master_endpoint)
    all_infos = exchange_service_info(name, rank, world_size, server_endpoint,
                                      master_endpoint)
    # init service info
    infos = []
    for node_info in all_infos:
        ip, port = node_info.endpoint.split(":")
        port = int(port)
        info = core.ServiceInfo(node_info.name, node_info.rank, ip, port)
        infos.append(info)
    agent = core.RpcAgent(name, infos)
    core.set_agent_instance(agent)
    agent.start_server()
    barrier(rank, world_size, master_endpoint)
    agent.start_client()


def rpc_sync(name, fn, args=None, kwargs=None):
    fut = _invoke_rpc(name, fn, args, kwargs)
    return deserialize(fut.wait())


def rpc_async(name, fn, args=None, kwargs=None):
    return _invoke_rpc(name, fn, args, kwargs)


def _invoke_rpc(name, fn, args, kwargs):
    args = args if args else ()
    kwargs = kwargs if kwargs else {}
    serial_obj = serialize(PythonFunc(fn, args, kwargs))
    future = core.invoke_rpc(name, serial_obj)
    return future


def shutdown():
    rank = core.rpc_get_rank()
    world_size = core.rpc_get_world_size()
    end_point = "{}:{}".format(MASTER_ADDR, MASTER_PORT)
    barrier(rank, world_size, end_point)
    core.rpc_stop_server()
    core.rpc_clear_python_rpc_handler()


def get_service_info(name):
    return core.rpc_get_service_info(name)


def get_all_service_infos():
    return core.rpc_get_all_service_infos()


def get_current_service_info():
    return core.rpc_get_current_service_info()
