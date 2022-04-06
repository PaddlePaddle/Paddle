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

from .device import Device

import socket
import struct
from contextlib import closing


class Node(object):
    def __init__(self):
        # self.device = Device.detect_device()
        self.device = Device.parse_device()
        self.ip = self.get_host_ip()
        self.free_ports = []
        self._allocated_ports = []

    def get_host_ip(self):
        try:
            self.hostname = socket.gethostname()
            self.ip = socket.gethostbyname(socket.getfqdn(self.hostname))
            return self.ip
        except:
            return '127.0.0.1'

    def get_free_ports(self, n=1):
        free_ports = [self.get_free_port() for i in range(n)]
        self.free_ports += free_ports
        return free_ports

    def get_ports_occupied(self):
        return self.free_ports

    def get_free_port(self):
        # for loop to avoid port conflict
        for _ in range(100):
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]
                if port in self._allocated_ports:
                    continue
                else:
                    self._allocated_ports.append(port)
                    return port
        return port

    @classmethod
    def is_server_ready(self, ip, port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            #sock.settimeout(0.01)
            #sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, 'SO_REUSEPORT'):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            result = sock.connect_ex((ip, int(port)))
            if result == 0:
                return True
            else:
                return False
