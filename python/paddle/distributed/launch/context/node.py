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

import os
import random
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

        port_range = os.getenv('PORT_RANGE', '35100:64000')
        port_range = port_range.split(':')
        self._port_start = int(port_range[0])
        self._port_end = int(port_range[1])
        self._port_cur = random.randint(self._port_start, self._port_end)

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

    def _get_free_port(self, port=0):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                         struct.pack('ii', 1, 0))
            try:
                s.bind(('', port))
                return s.getsockname()[1]
            except:
                return -1

    def _update_port_cur(self):
        self._port_cur += 1
        if self._port_cur > self._port_end:
            self._port_cur = self._port_start

    def get_free_port(self):
        for _ in range(100):
            ret = self._get_free_port(self._port_cur)
            if ret > 0:
                self._update_port_cur()
                return ret
            else:
                self._update_port_cur()

        return self._port_cur

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
