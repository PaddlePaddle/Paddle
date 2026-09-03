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

# [AUTO-GENERATED] Test file for paddle.distributed.launch.utils.topology
# Target file: paddle/distributed/launch/utils/topology.py
# 覆盖模块: paddle/distributed/launch/utils/topology.py
# Covered module: paddle/distributed/launch/utils/topology.py

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from paddle.distributed.launch.utils.topology import (
    SingleNodeTopology,
    call_cmd,
)

# Get module reference for patching
_topo_mod = sys.modules['paddle.distributed.launch.utils.topology']


class TestCallCmd(unittest.TestCase):
    """测试 call_cmd 函数 / Test call_cmd function"""

    @patch('subprocess.Popen')
    def test_call_cmd_success(self, mock_popen):
        """测试成功执行命令 / Test successful command execution"""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ('output_line\n', '')
        mock_popen.return_value = mock_proc
        result = call_cmd('echo hello', 'error', 'default')
        self.assertEqual(result, 'output_line\n')

    @patch('subprocess.Popen')
    def test_call_cmd_with_stderr(self, mock_popen):
        """测试命令有 stderr 时返回默认值 / Test command with stderr returns default"""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ('', 'some error')
        mock_popen.return_value = mock_proc
        result = call_cmd('bad_cmd', 'error msg', 'default_val')
        self.assertEqual(result, 'default_val')

    @patch('subprocess.Popen')
    def test_call_cmd_empty_stdout_no_stderr(self, mock_popen):
        """测试命令无输出且无错误 / Test command with empty stdout and no stderr"""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ('', '')
        mock_popen.return_value = mock_proc
        result = call_cmd('true', 'error', 'fallback')
        # When no stderr, returns stdout as-is (empty string)
        self.assertEqual(result, '')

    @patch('subprocess.Popen')
    def test_call_cmd_multiline_output(self, mock_popen):
        """测试命令多行输出 / Test command with multiline output"""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ('line1\nline2\nline3', '')
        mock_popen.return_value = mock_proc
        result = call_cmd('cat file', 'error', '')
        self.assertEqual(result, 'line1\nline2\nline3')


class TestSingleNodeTopologyInit(unittest.TestCase):
    """测试 SingleNodeTopology 初始化 / Test SingleNodeTopology initialization"""

    def test_init_defaults(self):
        """测试默认初始化值 / Test default initialization values"""
        topo = SingleNodeTopology()
        self.assertEqual(topo.pcie_latency, 0.0)
        self.assertEqual(topo.pcie_bandwidth, float('inf'))
        self.assertEqual(topo.nvlink_bandwidth, -1.0)
        self.assertEqual(topo.nb_devices, 8)
        self.assertEqual(topo.machine, {})
        self.assertEqual(topo.devices, [])
        self.assertEqual(topo.links, [])
        self.assertIsNone(topo.json_object)


class TestSingleNodeTopologyPCIeBandwidth(unittest.TestCase):
    """测试 pcie_gen2bandwidth 方法 / Test pcie_gen2bandwidth method"""

    def test_pcie_gen1(self):
        """测试 PCIe Gen1 带宽 / Test PCIe Gen1 bandwidth"""
        topo = SingleNodeTopology()
        self.assertEqual(topo.pcie_gen2bandwidth(1), 0.25)

    def test_pcie_gen2(self):
        """测试 PCIe Gen2 带宽 / Test PCIe Gen2 bandwidth"""
        topo = SingleNodeTopology()
        self.assertEqual(topo.pcie_gen2bandwidth(2), 0.5)

    def test_pcie_gen3(self):
        """测试 PCIe Gen3 带宽 / Test PCIe Gen3 bandwidth"""
        topo = SingleNodeTopology()
        self.assertEqual(topo.pcie_gen2bandwidth(3), 1.0)

    def test_pcie_gen4(self):
        """测试 PCIe Gen4 带宽 / Test PCIe Gen4 bandwidth"""
        topo = SingleNodeTopology()
        self.assertEqual(topo.pcie_gen2bandwidth(4), 2.0)

    def test_pcie_gen5(self):
        """测试 PCIe Gen5 带宽 / Test PCIe Gen5 bandwidth"""
        topo = SingleNodeTopology()
        self.assertEqual(topo.pcie_gen2bandwidth(5), 4.0)

    def test_pcie_gen6(self):
        """测试 PCIe Gen6 带宽 / Test PCIe Gen6 bandwidth"""
        topo = SingleNodeTopology()
        self.assertEqual(topo.pcie_gen2bandwidth(6), 8.0)

    def test_pcie_unknown_gen(self):
        """测试未知 PCIe 代 / Test unknown PCIe generation"""
        topo = SingleNodeTopology()
        result = topo.pcie_gen2bandwidth(99)
        self.assertIsNone(result)


class TestSingleNodeTopologyModel2GFlops(unittest.TestCase):
    """测试 model2gflops 方法 / Test model2gflops method"""

    def test_h100_sxm5(self):
        """测试 H100 SXM5 GFLOPS / Test H100 SXM5 GFLOPS"""
        topo = SingleNodeTopology()
        sp, dp = topo.model2gflops("H100 SXM5")
        self.assertEqual(sp, 60000)
        self.assertEqual(dp, 30000)

    def test_h100_pcie(self):
        """测试 H100 PCIe GFLOPS / Test H100 PCIe GFLOPS"""
        topo = SingleNodeTopology()
        sp, dp = topo.model2gflops("H100 PCIe")
        self.assertEqual(sp, 48000)
        self.assertEqual(dp, 24000)

    def test_a100(self):
        """测试 A100 GFLOPS / Test A100 GFLOPS"""
        topo = SingleNodeTopology()
        sp, dp = topo.model2gflops("NVIDIA A100-SXM4-40GB")
        self.assertEqual(sp, 19500)
        self.assertEqual(dp, 9700)

    def test_a800(self):
        """测试 A800 GFLOPS / Test A800 GFLOPS"""
        topo = SingleNodeTopology()
        sp, dp = topo.model2gflops("A800")
        self.assertEqual(sp, 19500)
        self.assertEqual(dp, 9700)

    def test_v100(self):
        """测试 V100 GFLOPS / Test V100 GFLOPS"""
        topo = SingleNodeTopology()
        sp, dp = topo.model2gflops("V100")
        self.assertEqual(sp, 15700)
        self.assertEqual(dp, 7800)

    def test_p100(self):
        """测试 P100 GFLOPS / Test P100 GFLOPS"""
        topo = SingleNodeTopology()
        sp, dp = topo.model2gflops("P100")
        self.assertEqual(sp, 10600)
        self.assertEqual(dp, 5300)

    def test_unknown_model(self):
        """测试未知模型返回 None / Test unknown model returns None"""
        topo = SingleNodeTopology()
        result = topo.model2gflops("UnknownGPU")
        self.assertIsNone(result)


class TestSingleNodeTopologyCalculateCPUFlops(unittest.TestCase):
    """测试 calculate_cpu_flops 方法 / Test calculate_cpu_flops method"""

    @patch.object(_topo_mod, 'call_cmd')
    def test_calculate_cpu_flops_sse(self, mock_call):
        """测试 SSE 指令集的 CPU FLOPS 计算
        Test CPU FLOPS calculation with SSE"""

        def side_effect(cmd, err, default):
            if 'Socket(s)' in cmd:
                return '4'
            elif 'Core(s) per socket' in cmd:
                return '20'
            elif 'GHz' in cmd:
                return '2.4'
            elif 'grep sse' in cmd:
                return 'sse4_2'
            elif 'grep avx2' in cmd:
                return ''
            elif 'grep avx512' in cmd:
                return ''
            return default

        mock_call.side_effect = side_effect
        topo = SingleNodeTopology()
        topo.calculate_cpu_flops()
        self.assertIn('sp_gflops', topo.machine)
        self.assertIn('dp_gflops', topo.machine)

    @patch.object(_topo_mod, 'call_cmd')
    def test_calculate_cpu_flops_avx512(self, mock_call):
        """测试 AVX512 指令集的 CPU FLOPS
        Test CPU FLOPS with AVX512"""
        mock_call.side_effect = [
            '2',  # sockets
            '24',  # cores per socket
            '3.0',  # clock rate
            'avx512f',  # sse
            'avx512f',  # avx2
            'avx512f',  # avx512
        ]
        topo = SingleNodeTopology()
        topo.calculate_cpu_flops()
        self.assertIn('sp_gflops', topo.machine)
        # AVX512: sp = gflops_per_element * 16, dp = gflops_per_element * 8
        sp_gflops = topo.machine['sp_gflops']
        dp_gflops = topo.machine['dp_gflops']
        self.assertEqual(sp_gflops, 2 * dp_gflops)

    @patch.object(_topo_mod, 'call_cmd')
    def test_calculate_cpu_flops_no_simd(self, mock_call):
        """测试无 SIMD 指令集的 CPU FLOPS
        Test CPU FLOPS with no SIMD"""
        mock_call.side_effect = [
            '2',
            '20',
            '2.4',
            '',  # no sse
            '',  # no avx2
            '',  # no avx512
        ]
        topo = SingleNodeTopology()
        topo.calculate_cpu_flops()
        # No SIMD means width is 0
        self.assertEqual(topo.machine['sp_gflops'], 0)
        self.assertEqual(topo.machine['dp_gflops'], 0)


class TestSingleNodeTopologyDump(unittest.TestCase):
    """测试 dump 方法 / Test dump method"""

    def test_dump(self):
        """测试 dump 到文件 / Test dump to file"""
        topo = SingleNodeTopology()
        topo.machine['hostname'] = 'test_host'
        topo.machine['memory'] = 64
        with tempfile.NamedTemporaryFile(
            suffix='.json', delete=False, mode='w'
        ) as f:
            output_path = f.name
        try:
            topo.dump(output_path)
            with open(output_path, 'r') as f:
                data = json.load(f)
            self.assertEqual(data['hostname'], 'test_host')
            self.assertEqual(data['memory'], 64)
        finally:
            os.unlink(output_path)


class TestGetHostInfo(unittest.TestCase):
    """测试 get_host_info 方法 / Test get_host_info method"""

    @patch.object(_topo_mod, 'call_cmd')
    def test_get_host_info(self, mock_call):
        """测试获取主机信息 / Test getting host info"""

        def side_effect(cmd, err, default):
            if 'hostname -s' in cmd:
                return 'test-host\n'
            elif 'hostname -i' in cmd:
                return '192.168.1.1\n'
            elif 'MemAvailable' in cmd:
                return '41366484\n'
            elif 'Socket(s)' in cmd:
                return '4\n'
            elif 'Core(s) per socket' in cmd:
                return '20\n'
            elif 'GHz' in cmd:
                return '2.4\n'
            elif 'grep sse' in cmd:
                return 'sse\n'
            elif 'grep avx2' in cmd:
                return 'avx2\n'
            elif 'grep avx512' in cmd:
                return 'avx512\n'
            return default

        mock_call.side_effect = side_effect
        topo = SingleNodeTopology()
        topo.get_host_info()
        self.assertEqual(topo.machine['hostname'], 'test-host')
        self.assertEqual(topo.machine['addr'], '192.168.1.1')
        self.assertEqual(topo.machine['memory'], 41)

    @patch.object(_topo_mod, 'call_cmd')
    def test_get_host_info_with_errors(self, mock_call):
        """测试命令出错时获取主机信息 / Test getting host info with command errors"""

        def side_effect(cmd, err, default):
            if 'hostname -s' in cmd:
                return 'localhost\n'
            elif 'hostname -i' in cmd:
                return '127.0.0.1\n'
            elif 'MemAvailable' in cmd:
                return '41366484\n'
            elif 'Socket(s)' in cmd:
                return '4\n'
            elif 'Core(s) per socket' in cmd:
                return '20\n'
            elif 'GHz' in cmd:
                return '2.4\n'
            elif 'sse' in cmd:
                return ''
            elif 'avx2' in cmd:
                return ''
            elif 'avx512' in cmd:
                return ''
            return default

        mock_call.side_effect = side_effect
        topo = SingleNodeTopology()
        topo.get_host_info()
        self.assertEqual(topo.machine['hostname'], 'localhost')
        self.assertEqual(topo.machine['addr'], '127.0.0.1')


class TestGetLinkBandwidth(unittest.TestCase):
    """测试 get_link_bandwidth 方法 / Test get_link_bandwidth method"""

    @patch.object(_topo_mod, 'call_cmd')
    def test_get_link_bandwidth_nvlink(self, mock_call):
        """测试 NVLink 带宽 / Test NVLink bandwidth"""
        mock_call.side_effect = [
            'NV4\n',  # link type
            '25\n',  # nvlink bandwidth
        ]
        topo = SingleNodeTopology()
        topo.nvlink_bandwidth = -1.0
        link_type, bw = topo.get_link_bandwidth(0, 1)
        self.assertEqual(link_type, 'NVL')
        self.assertEqual(bw, 100.0)  # 4 * 25

    @patch.object(_topo_mod, 'call_cmd')
    def test_get_link_bandwidth_pcie(self, mock_call):
        """测试 PCIe 带宽 / Test PCIe bandwidth"""
        mock_call.side_effect = ['PIX\n']
        topo = SingleNodeTopology()
        topo.pcie_bandwidth = 32.0
        link_type, bw = topo.get_link_bandwidth(0, 1)
        self.assertIn('PIX', link_type)
        self.assertEqual(bw, 32.0)

    @patch.object(_topo_mod, 'call_cmd')
    def test_get_link_bandwidth_nvlink_cached(self, mock_call):
        """测试 NVLink 带宽缓存 / Test NVLink bandwidth cached"""
        mock_call.side_effect = [
            'NV2\n',  # link type
            '50\n',  # nvlink bandwidth
        ]
        topo = SingleNodeTopology()
        topo.nvlink_bandwidth = -1.0
        link_type, bw = topo.get_link_bandwidth(0, 1)
        self.assertEqual(topo.nvlink_bandwidth, 50.0)
        # Second call should use cached value
        mock_call.side_effect = [
            'NV2\n',
            '25\n',  # this should not be used since cached
        ]
        link_type2, bw2 = topo.get_link_bandwidth(1, 2)
        # The cached nvlink_bandwidth is still 50.0
        self.assertEqual(bw2, 100.0)  # 2 * 50


if __name__ == '__main__':
    unittest.main()
