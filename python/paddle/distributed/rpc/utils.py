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

import socket
import pickle
import time
from collections import namedtuple
from contextlib import closing

ServiceInfo = namedtuple("ServiceInfo", ["name", "rank", "endpoint"])


def _exchange_service_info(name, rank, world_size, current_endpoint,
                           master_endpoint):
    master_addr, master_port = master_endpoint.split(":")
    master_port = int(master_port)
    # compute the size of data to be received
    tmp = ServiceInfo(name, rank, current_endpoint)
    tmp = pickle.dumps(tmp)
    master_size = len(tmp) + 1024
    children_size = len(tmp) * world_size + 1024
    if rank == 0:
        master_info = ServiceInfo(name, rank, current_endpoint)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # set port reuse, related issue: https://lienze.tech/archives/149/
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((master_addr, master_port))
        server.listen(5)
        server.setblocking(True)
        count = 1
        all_conn = []
        # 1. build connection
        while count != world_size:
            conn, _ = server.accept()
            all_conn.append(conn)
            count += 1
        all_service_info = [master_info]
        # 2. client send service info to root
        for conn in all_conn:
            client_data = conn.recv(master_size)
            client_data = pickle.loads(client_data)
            all_service_info.append(client_data)
        # 3. root send all service infos to client
        for conn in all_conn:
            msg = all_service_info
            msg = pickle.dumps(msg)
            conn.send(msg)
        for conn in all_conn:
            conn.close()
        server.close()
        return all_service_info
    else:
        with closing(socket.socket(socket.AF_INET,
                                   socket.SOCK_STREAM)) as client:
            client.settimeout(2)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            result = -1
            while result != 0:
                result = client.connect_ex((master_addr, master_port))
                if result != 0:
                    time.sleep(0.1)
            msg = ServiceInfo(name, rank, current_endpoint)
            msg = pickle.dumps(msg)
            client.send(msg)
            data = client.recv(children_size)
            data = pickle.loads(data)
        return data


def _barrier(rank, world_size, master_endpoint):
    """
    When all servers have executed here, they can continue to execute.
    reference: https://stackoverflow.com/questions/21626423/how-is-barrier-implemented-in-message-passing-systems
    """
    master_addr, master_port = master_endpoint.split(":")
    master_port = int(master_port)
    if rank == 0:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((master_addr, master_port))
        server.listen(5)
        server.setblocking(True)
        count = 1
        all_conn = []
        while count != world_size:
            conn, _ = server.accept()
            all_conn.append(conn)
            count += 1
        for conn in all_conn:
            conn.recv(1024).decode()
        for conn in all_conn:
            msg = "success"
            conn.send(msg.encode("utf-8"))
        for conn in all_conn:
            conn.close()
        server.close()
    else:
        with closing(socket.socket(socket.AF_INET,
                                   socket.SOCK_STREAM)) as client:
            client.settimeout(2)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            result = -1
            while result != 0:
                result = client.connect_ex((master_addr, master_port))
                if result != 0:
                    time.sleep(0.1)
            msg = "hello world"
            client.send(msg.encode("utf-8"))
            client.recv(1024).decode()
