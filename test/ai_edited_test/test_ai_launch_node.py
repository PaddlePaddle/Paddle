# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Test file for paddle.distributed.launch.context.node
# Target file: paddle/distributed/launch/context/node.py
# 覆盖模块: paddle/distributed/launch/context/node.py
# Covered module: paddle/distributed/launch/context/node.py

import os
import socket
import sys
import unittest
from unittest.mock import MagicMock, patch

from paddle.distributed.launch.context.node import Node

# Get module references for patching (dotted path doesn't work for launch module)
_node_mod = sys.modules['paddle.distributed.launch.context.node']
_device_mod = sys.modules['paddle.distributed.launch.context.device']


class TestNodeInit(unittest.TestCase):
    """测试 Node 初始化 / Test Node initialization"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_node_init_default(self, mock_parse):
        """测试 Node 默认初始化 / Test Node default initialization"""
        mock_dev = MagicMock()
        mock_dev.dtype = 'cpu'
        mock_parse.return_value = mock_dev
        node = Node()
        self.assertIsNotNone(node)
        self.assertIsNotNone(node.ip)
        self.assertEqual(node.free_ports, [])
        self.assertEqual(node._allocated_ports, [])

    @patch.object(_device_mod.Device, 'parse_device')
    def test_node_init_port_range(self, mock_parse):
        """测试 Node 端口范围初始化 / Test Node port range initialization"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        env_backup = os.environ.get('PORT_RANGE')
        os.environ['PORT_RANGE'] = '40000:50000'
        try:
            node = Node()
            self.assertEqual(node._port_start, 40000)
            self.assertEqual(node._port_end, 50000)
        finally:
            if env_backup is not None:
                os.environ['PORT_RANGE'] = env_backup
            else:
                del os.environ['PORT_RANGE']


class TestNodeGetHostIp(unittest.TestCase):
    """测试 Node.get_host_ip / Test Node.get_host_ip"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_host_ip_success(self, mock_parse):
        """测试成功获取主机 IP / Test successfully getting host IP"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        self.assertIsNotNone(node.ip)
        self.assertIsInstance(node.ip, str)

    @patch.object(_device_mod.Device, 'parse_device')
    @patch('socket.gethostbyname', side_effect=socket.gaierror('error'))
    def test_get_host_ip_failure(self, mock_getbyname, mock_parse):
        """测试获取主机 IP 失败时返回 127.0.0.1
        Test getting host IP failure returns 127.0.0.1"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        self.assertEqual(node.ip, '127.0.0.1')


class TestNodeGetFreePorts(unittest.TestCase):
    """测试 Node.get_free_ports / Test Node.get_free_ports"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_ports_single(self, mock_parse):
        """测试获取单个空闲端口 / Test getting a single free port"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        ports = node.get_free_ports(1)
        self.assertEqual(len(ports), 1)
        self.assertIsInstance(ports[0], int)

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_ports_multiple(self, mock_parse):
        """测试获取多个空闲端口 / Test getting multiple free ports"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        ports = node.get_free_ports(3)
        self.assertEqual(len(ports), 3)
        # All ports should be unique
        self.assertEqual(len(set(ports)), 3)

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_ports_accumulates(self, mock_parse):
        """测试获取的端口累计到 free_ports / Test ports accumulate in free_ports"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        ports1 = node.get_free_ports(2)
        ports2 = node.get_free_ports(3)
        self.assertEqual(len(node.free_ports), 5)

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_ports_with_fixed_port(self, mock_parse):
        """测试使用固定端口 / Test getting ports with fixed port"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        env_backup = os.environ.get('FLAGS_FIXED_PORT')
        os.environ['FLAGS_FIXED_PORT'] = '12345'
        try:
            node = Node()
            ports = node.get_free_ports(3, rank=0)
            self.assertEqual(ports, [12345, 12346, 12347])
        finally:
            if env_backup is not None:
                os.environ['FLAGS_FIXED_PORT'] = env_backup
            else:
                del os.environ['FLAGS_FIXED_PORT']

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_ports_with_fixed_port_rank(self, mock_parse):
        """测试固定端口带 rank / Test fixed ports with rank offset"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        env_backup = os.environ.get('FLAGS_FIXED_PORT')
        os.environ['FLAGS_FIXED_PORT'] = '20000'
        try:
            node = Node()
            ports = node.get_free_ports(2, rank=3)
            self.assertEqual(ports, [20003, 20004])
        finally:
            if env_backup is not None:
                os.environ['FLAGS_FIXED_PORT'] = env_backup
            else:
                del os.environ['FLAGS_FIXED_PORT']


class TestNodeGetPortsOccupied(unittest.TestCase):
    """测试 Node.get_ports_occupied / Test Node.get_ports_occupied"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_ports_occupied_empty(self, mock_parse):
        """测试无占用端口 / Test no occupied ports"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        self.assertEqual(node.get_ports_occupied(), [])

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_ports_occupied_after_allocate(self, mock_parse):
        """测试分配后占用端口 / Test occupied ports after allocation"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        ports = node.get_free_ports(2)
        occupied = node.get_ports_occupied()
        self.assertEqual(occupied, ports)


class TestNodeGetFreePort(unittest.TestCase):
    """测试 Node.get_free_port / Test Node.get_free_port"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_port(self, mock_parse):
        """测试获取单个空闲端口 / Test getting a single free port"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        port = node.get_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)


class TestNodeUpdatePortCur(unittest.TestCase):
    """测试 Node._update_port_cur / Test Node._update_port_cur"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_update_port_cur_increments(self, mock_parse):
        """测试端口游标递增 / Test port cursor increments"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        env_backup = os.environ.get('PORT_RANGE')
        os.environ['PORT_RANGE'] = '40000:40100'
        try:
            node = Node()
            old_cur = node._port_cur
            node._update_port_cur()
            # If old_cur was 40099, it wraps to 40000
            if old_cur >= 40100:
                self.assertEqual(node._port_cur, 40000)
            else:
                self.assertEqual(node._port_cur, old_cur + 1)
        finally:
            if env_backup is not None:
                os.environ['PORT_RANGE'] = env_backup
            else:
                del os.environ['PORT_RANGE']

    @patch.object(_device_mod.Device, 'parse_device')
    def test_update_port_cur_wraps(self, mock_parse):
        """测试端口游标回绕 / Test port cursor wraps around"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        env_backup = os.environ.get('PORT_RANGE')
        os.environ['PORT_RANGE'] = '40000:40010'
        try:
            node = Node()
            node._port_cur = 40010
            node._update_port_cur()
            self.assertEqual(node._port_cur, 40000)
        finally:
            if env_backup is not None:
                os.environ['PORT_RANGE'] = env_backup
            else:
                del os.environ['PORT_RANGE']


class TestNodeGetFreePortMethod(unittest.TestCase):
    """测试 Node._get_free_port 内部方法 / Test Node._get_free_port internal method"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_port_specific(self, mock_parse):
        """测试指定端口 / Test getting specific port"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        port = node._get_free_port(0)  # Port 0 means OS assigns
        self.assertGreater(port, 0)

    @patch.object(_device_mod.Device, 'parse_device')
    def test_get_free_port_in_use(self, mock_parse):
        """测试端口已占用返回 -1 / Test port in use returns -1"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        # Bind a port first
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        try:
            # Try to get that same port (should fail because it's bound)
            result = node._get_free_port(port)
            self.assertEqual(result, -1)
        finally:
            sock.close()


class TestNodeIsServerReady(unittest.TestCase):
    """测试 Node.is_server_ready / Test Node.is_server_ready"""

    def test_is_server_ready_no_server(self):
        """测试无服务器时返回 False / Test returns False when no server"""
        result = Node.is_server_ready('127.0.0.1', 12345)
        self.assertFalse(result)

    def test_is_server_ready_with_server(self):
        """测试有服务器时返回 True / Test returns True when server is ready"""
        # Create a listening server
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('127.0.0.1', 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]
        try:
            result = Node.is_server_ready('127.0.0.1', port)
            self.assertTrue(result)
        finally:
            server_sock.close()

    def test_is_server_ready_port_string(self):
        """测试端口号为字符串 / Test with port as string"""
        result = Node.is_server_ready('127.0.0.1', '12345')
        self.assertFalse(result)


class TestNodeGetHostIpHostname(unittest.TestCase):
    """测试 Node.get_host_ip 主机名 / Test Node.get_host_ip hostname"""

    @patch.object(_device_mod.Device, 'parse_device')
    def test_hostname_attribute(self, mock_parse):
        """测试 Node 设置 hostname 属性 / Test Node sets hostname attribute"""
        mock_dev = MagicMock()
        mock_parse.return_value = mock_dev
        node = Node()
        self.assertTrue(hasattr(node, 'hostname'))
        self.assertIsNotNone(node.hostname)


if __name__ == '__main__':
    unittest.main()
