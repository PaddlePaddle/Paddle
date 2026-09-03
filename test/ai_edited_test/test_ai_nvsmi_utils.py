# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.distributed.launch.utils.nvsmi
# 自动生成的单测，覆盖 nvsmi 模块中未覆盖的代码
# Target: cover uncovered lines 28,31,34,38,53,61,66,82-111,115-150,154-185
#   in python/paddle/distributed/launch/utils/nvsmi.py
# 未覆盖行: Info.json/str, query_smi return None 分支, query_rocm_smi, query_npu_smi,
#           query_xpu_smi, has_*_smi 函数

import importlib
import json
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch

# Skip tests in environments where nvidia-smi is unavailable (e.g. CI Docker)
# 在没有 nvidia-smi 的环境（如 CI Docker）中跳过测试
_NVIDIA_SMI_AVAILABLE = shutil.which("nvidia-smi") is not None

# Ensure module is imported before accessing sys.modules
# 确保在访问 sys.modules 之前先导入模块
# Gracefully skip if the module is unavailable in some CI environments
# 在某些 CI 环境中模块不可用时优雅跳过
_MODULE_AVAILABLE = False
nvsmi_mod = None
try:
    importlib.import_module('paddle.distributed.launch.utils.nvsmi')
    nvsmi_mod = sys.modules.get('paddle.distributed.launch.utils.nvsmi')
    if nvsmi_mod is not None:
        _MODULE_AVAILABLE = True
except (ImportError, KeyError, AttributeError):
    pass

if _MODULE_AVAILABLE:
    from paddle.distributed.launch.utils.nvsmi import (
        Info,
        get_gpu_info,
        get_gpu_process,
        get_gpu_util,
        has_npu_smi,
        has_nvidia_smi,
        has_rocm_smi,
        has_xpu_smi,
        query_npu_smi,
        query_rocm_smi,
        query_smi,
        query_xpu_smi,
    )

    NPS = nvsmi_mod.__name__


_SMI_SKIP = unittest.skipUnless(
    _NVIDIA_SMI_AVAILABLE and _MODULE_AVAILABLE,
    "nvidia-smi not available or nvsmi module not importable",
)


@_SMI_SKIP
class TestInfo(unittest.TestCase):
    """Test the Info helper class.
    测试 Info 辅助类。"""

    def test_repr(self):
        """Info.__repr__ should return string of __dict__.
        Info.__repr__ 应返回 __dict__ 的字符串表示。"""
        info = Info()
        info.name = "GPU0"
        info.memory = "8192"
        result = repr(info)
        self.assertIn("name", result)
        self.assertIn("GPU0", result)

    def test_json(self):
        """Info.json() should return JSON string.
        Info.json() 应返回 JSON 字符串。"""
        info = Info()
        info.index = 0
        info.utilization = 80
        result = info.json()
        parsed = json.loads(result)
        self.assertEqual(parsed["index"], 0)
        self.assertEqual(parsed["utilization"], 80)

    def test_dict(self):
        """Info.dict() should return __dict__.
        Info.dict() 应返回 __dict__。"""
        info = Info()
        info.key = "value"
        result = info.dict()
        self.assertEqual(result, {"key": "value"})

    def test_str_no_keys(self):
        """Info.str() with no keys returns all values joined.
        Info.str() 不带 keys 参数时返回所有值拼接。"""
        info = Info()
        info.a = "1"
        info.b = "2"
        result = info.str()
        self.assertEqual(result, "1,2")

    def test_str_with_keys_list(self):
        """Info.str() with keys list filters values.
        Info.str() 带 keys 列表时过滤值。"""
        info = Info()
        info.a = "x"
        info.b = "y"
        info.c = "z"
        result = info.str(keys=["a", "c"])
        self.assertEqual(result, "x,z")

    def test_str_with_keys_string(self):
        """Info.str() with comma-separated string keys.
        Info.str() 带逗号分隔的字符串 keys。"""
        info = Info()
        info.index = 0
        info.name = "gpu"
        info.util = 50
        result = info.str(keys="index,util")
        self.assertEqual(result, "0,50")

    def test_str_missing_key(self):
        """Info.str() with missing key returns empty string for that key.
        Info.str() 遇到缺失的 key 时返回空字符串。"""
        info = Info()
        info.a = "1"
        result = info.str(keys=["a", "nonexistent"])
        self.assertEqual(result, "1,")


@_SMI_SKIP
class TestQuerySmi(unittest.TestCase):
    """Test query_smi function.
    测试 query_smi 函数。"""

    def test_query_smi_no_nvidia_smi(self):
        """query_smi returns empty list when nvidia-smi not found.
        当找不到 nvidia-smi 时，query_smi 返回空列表。"""
        with patch.object(nvsmi_mod, "has_nvidia_smi", return_value=False):
            result = query_smi(query=["index", "name"])
            self.assertEqual(result, [])

    def test_query_smi_none_query_type(self):
        """query_smi returns None when query is None (else branch).
        当 query 为 None 时，query_smi 返回 None（else 分支）。"""
        with patch.object(nvsmi_mod, "has_nvidia_smi", return_value=True):
            result = query_smi(query=None)
            self.assertIsNone(result)

    def test_query_smi_gpu_query(self):
        """query_smi with GPU query parses output correctly.
        使用 GPU 查询的 query_smi 正确解析输出。"""
        with (
            patch.object(nvsmi_mod, "has_nvidia_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"0, Tesla V100, 8192\n1, Tesla K80, 4096\n",
            ),
        ):
            result = query_smi(
                query=["index", "name", "memory"],
                query_type="gpu",
                dtype=[int, str, int],
            )
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].index, 0)
            self.assertEqual(result[0].name, "Tesla V100")
            self.assertEqual(result[0].memory, 8192)

    def test_query_smi_with_index(self):
        """query_smi with index filter.
        带索引过滤的 query_smi。"""
        with (
            patch.object(nvsmi_mod, "has_nvidia_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"0, 50, 8192\n",
            ),
        ):
            result = query_smi(
                query=["index", "utilization.gpu", "memory.total"],
                query_type="gpu",
                index=["0"],
                dtype=[int, int, int],
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].index, 0)

    def test_query_smi_compute_query(self):
        """query_smi with compute query type.
        使用 compute 查询类型的 query_smi。"""
        with (
            patch.object(nvsmi_mod, "has_nvidia_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"1234, python, GPU-xxx, 1024\n",
            ),
        ):
            result = query_smi(
                query=["pid", "process_name", "gpu_uuid", "used_memory"],
                query_type="compute",
                dtype=[int, str, str, int],
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].pid, 1234)

    def test_query_smi_default_dtype(self):
        """query_smi uses str dtype when not matching query length.
        query_smi 在 dtype 不匹配查询长度时使用 str 类型。"""
        with (
            patch.object(nvsmi_mod, "has_nvidia_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess, "check_output", return_value=b"0, Tesla\n"
            ),
        ):
            result = query_smi(query=["index", "name"], query_type="gpu")
            # dtype defaults to [str, str] when not provided (line 66)
            # 未提供时 dtype 默认为 [str, str]（第66行）
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].index, "0")
            self.assertEqual(result[0].name, "Tesla")

    def test_query_smi_empty_lines(self):
        """query_smi skips empty lines.
        query_smi 跳过空行。"""
        with (
            patch.object(nvsmi_mod, "has_nvidia_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"0, Tesla\n\n\n",
            ),
        ):
            result = query_smi(
                query=["index", "name"], query_type="gpu", dtype=[int, str]
            )
            self.assertEqual(len(result), 1)


@_SMI_SKIP
class TestQueryRocmSmi(unittest.TestCase):
    """Test query_rocm_smi function.
    测试 query_rocm_smi 函数。"""

    def test_query_rocm_smi_not_found(self):
        """query_rocm_smi returns empty list when rocm-smi not found.
        当找不到 rocm-smi 时，query_rocm_smi 返回空列表。"""
        with patch.object(nvsmi_mod, "has_rocm_smi", return_value=False):
            result = query_rocm_smi(query=["index", "name"])
            self.assertEqual(result, [])

    def test_query_rocm_smi_valid_output(self):
        """query_rocm_smi parses rocm-smi output.
        query_rocm_smi 解析 rocm-smi 输出。"""
        # Simulate rocm-smi output: 8 tokens, no DCU, with percentage
        # The function transforms line into 6 items:
        # [line[0], line[7][:-1], mem, mem*float(line[6][:-1])/100, mem-..., timestamp]
        # 模拟 rocm-smi 输出：8个令牌，无DCU，带百分比
        # 函数将行转换为6项：[line[0], line[7][:-1], mem, ...]
        with (
            patch.object(nvsmi_mod, "has_rocm_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"0    45C  50%  80%  32150M  50.0%  16075M  16075M\n",
            ),
        ):
            result = query_rocm_smi(
                query=[
                    "index",
                    "temperature",
                    "mem_total",
                    "mem_used",
                    "mem_free",
                    "timestamp",
                ],
                dtype=[int, str, int, float, float, str],
                mem=32150,
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].index, 0)

    def test_query_rocm_smi_skip_dcu_line(self):
        """query_rocm_smi skips DCU lines.
        query_rocm_smi 跳过 DCU 行。"""
        with (
            patch.object(nvsmi_mod, "has_rocm_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"DCU    0    45C  50%  80%  32150M\n",
            ),
        ):
            result = query_rocm_smi(
                query=["index", "name"],
                dtype=[int, str],
            )
            self.assertEqual(len(result), 0)

    def test_query_rocm_smi_wrong_token_count(self):
        """query_rocm_smi skips lines without exactly 8 tokens.
        query_rocm_smi 跳过不恰好有8个令牌的行。"""
        with (
            patch.object(nvsmi_mod, "has_rocm_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess, "check_output", return_value=b"0    45C\n"
            ),
        ):
            result = query_rocm_smi(query=["index", "name"], dtype=[int, str])
            self.assertEqual(len(result), 0)

    def test_query_rocm_smi_empty_lines(self):
        """query_rocm_smi skips empty lines.
        query_rocm_smi 跳过空行。"""
        with (
            patch.object(nvsmi_mod, "has_rocm_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess, "check_output", return_value=b"\n\n"
            ),
        ):
            result = query_rocm_smi(query=["index"], dtype=[int])
            self.assertEqual(len(result), 0)


@_SMI_SKIP
class TestQueryNpuSmi(unittest.TestCase):
    """Test query_npu_smi function.
    测试 query_npu_smi 函数。"""

    def test_query_npu_smi_not_found(self):
        """query_npu_smi returns empty list when npu-smi not found.
        当找不到 npu-smi 时，query_npu_smi 返回空列表。"""
        with patch.object(nvsmi_mod, "has_npu_smi", return_value=False):
            result = query_npu_smi(query=["index", "name"])
            self.assertEqual(result, [])

    def test_query_npu_smi_valid_output(self):
        """query_npu_smi parses npu-smi output with 18-19 fields.
        query_npu_smi 解析18-19个字段的 npu-smi 输出。"""
        # Need exactly 18-19 items from re.split BEFORE filtering empties.
        # Use commas without spaces. The function then picks result[2], [5], [6]
        # and produces 6 items: [i, name, mem_total, mem_used, mem_free, timestamp]
        # 需要 re.split 产生恰好18-19项（过滤空字符串之前）。
        # 使用不带空格的逗号。
        with (
            patch.object(nvsmi_mod, "has_npu_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"0,1,npu-name,100,50.0,32.0,16.0,health,a,b,c,d,e,f,g,h,i,j,k\n",
            ),
        ):
            result = query_npu_smi(
                query=[
                    "index",
                    "name",
                    "mem_total",
                    "mem_used",
                    "mem_free",
                    "timestamp",
                ],
                dtype=[int, str, float, float, float, str],
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].index, 0)

    def test_query_npu_smi_skip_npu_header(self):
        """query_npu_smi skips lines containing NPU.
        query_npu_smi 跳过包含 NPU 的行。"""
        with (
            patch.object(nvsmi_mod, "has_npu_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"NPU    0    1    info\n",
            ),
        ):
            result = query_npu_smi(query=["index"], dtype=[int])
            self.assertEqual(len(result), 0)

    def test_query_npu_smi_wrong_field_count(self):
        """query_npu_smi skips lines with wrong field count.
        query_npu_smi 跳过字段数错误的行。"""
        with (
            patch.object(nvsmi_mod, "has_npu_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess,
                "check_output",
                return_value=b"0, 1, 2, 3\n",
            ),
        ):
            result = query_npu_smi(query=["index"], dtype=[int])
            self.assertEqual(len(result), 0)

    def test_query_npu_smi_empty_lines(self):
        """query_npu_smi skips empty lines.
        query_npu_smi 跳过空行。"""
        with (
            patch.object(nvsmi_mod, "has_npu_smi", return_value=True),
            patch.object(
                nvsmi_mod.subprocess, "check_output", return_value=b"\n\n"
            ),
        ):
            result = query_npu_smi(query=["index"], dtype=[int])
            self.assertEqual(len(result), 0)


@_SMI_SKIP
class TestQueryXpuSmi(unittest.TestCase):
    """Test query_xpu_smi function.
    测试 query_xpu_smi 函数。"""

    def test_query_xpu_smi_no_attribute(self):
        """query_xpu_smi returns empty when core lacks get_xpu_device_count.
        当 core 缺少 get_xpu_device_count 时，query_xpu_smi 返回空列表。"""
        mock_core = MagicMock(spec=[])  # Empty spec, no attributes
        with patch.object(nvsmi_mod, "core", mock_core):
            result = query_xpu_smi(query=["index"])
            self.assertEqual(result, [])

    def test_query_xpu_smi_zero_devices(self):
        """query_xpu_smi returns empty when no XPU devices.
        当没有 XPU 设备时，query_xpu_smi 返回空列表。"""
        with patch.object(nvsmi_mod, "core") as mock_core:
            mock_core.get_xpu_device_count.return_value = 0
            result = query_xpu_smi(query=["index"])
            self.assertEqual(result, [])

    def test_query_xpu_smi_with_devices(self):
        """query_xpu_smi returns info for each XPU device.
        query_xpu_smi 为每个 XPU 设备返回信息。"""
        with patch.object(nvsmi_mod, "core") as mock_core:
            mock_core.get_xpu_device_count.return_value = 2
            mock_core.get_xpu_device_utilization_rate.return_value = 75
            mock_core.get_xpu_device_total_memory.return_value = (
                16384 * 1024 * 1024
            )
            mock_core.get_xpu_device_used_memory.return_value = (
                8192 * 1024 * 1024
            )

            result = query_xpu_smi(
                query=[
                    "index",
                    "utilization",
                    "mem_total",
                    "mem_used",
                    "mem_free",
                    "timestamp",
                ],
                dtype=[int, int, float, float, float, str],
            )
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].index, 0)
            self.assertEqual(result[1].index, 1)

    def test_query_xpu_smi_default_index(self):
        """query_xpu_smi uses all devices when no index provided.
        未提供索引时，query_xpu_smi 使用所有设备。"""
        with patch.object(nvsmi_mod, "core") as mock_core:
            mock_core.get_xpu_device_count.return_value = 3
            mock_core.get_xpu_device_utilization_rate.return_value = 50
            mock_core.get_xpu_device_total_memory.return_value = (
                32768 * 1024 * 1024
            )
            mock_core.get_xpu_device_used_memory.return_value = 0

            result = query_xpu_smi(
                query=["index", "utilization"],
                dtype=[int, int],
            )
            # Should iterate over range(3)
            # 应遍历 range(3)
            self.assertEqual(len(result), 3)

    def test_query_xpu_smi_custom_index(self):
        """query_xpu_smi with custom index list.
        使用自定义索引列表的 query_xpu_smi。"""
        with patch.object(nvsmi_mod, "core") as mock_core:
            mock_core.get_xpu_device_count.return_value = 4
            mock_core.get_xpu_device_utilization_rate.return_value = 30
            mock_core.get_xpu_device_total_memory.return_value = (
                8192 * 1024 * 1024
            )
            mock_core.get_xpu_device_used_memory.return_value = (
                4096 * 1024 * 1024
            )

            result = query_xpu_smi(
                query=["index", "utilization"],
                dtype=[int, int],
                index=[1, 3],
            )
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].index, 1)
            self.assertEqual(result[1].index, 3)


@_SMI_SKIP
class TestHasSmiFunctions(unittest.TestCase):
    """Test has_*_smi helper functions.
    测试 has_*_smi 辅助函数。"""

    def test_has_nvidia_smi_found(self):
        """has_nvidia_smi returns True when nvidia-smi is found.
        当找到 nvidia-smi 时，has_nvidia_smi 返回 True。"""
        with patch.object(
            nvsmi_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"
        ):
            self.assertTrue(has_nvidia_smi())

    def test_has_nvidia_smi_not_found(self):
        """has_nvidia_smi returns False when nvidia-smi is not found.
        当找不到 nvidia-smi 时，has_nvidia_smi 返回 False。"""
        with patch.object(nvsmi_mod.shutil, "which", return_value=None):
            self.assertFalse(has_nvidia_smi())

    def test_has_rocm_smi_found(self):
        """has_rocm_smi returns True when rocm-smi is found.
        当找到 rocm-smi 时，has_rocm_smi 返回 True。"""
        with patch.object(
            nvsmi_mod.shutil, "which", return_value="/usr/bin/rocm-smi"
        ):
            self.assertTrue(has_rocm_smi())

    def test_has_rocm_smi_not_found(self):
        """has_rocm_smi returns False when rocm-smi is not found.
        当找不到 rocm-smi 时，has_rocm_smi 返回 False。"""
        with patch.object(nvsmi_mod.shutil, "which", return_value=None):
            self.assertFalse(has_rocm_smi())

    def test_has_npu_smi_found(self):
        """has_npu_smi returns True when npu-smi is found.
        当找到 npu-smi 时，has_npu_smi 返回 True。"""
        with patch.object(
            nvsmi_mod.shutil, "which", return_value="/usr/bin/npu-smi"
        ):
            self.assertTrue(has_npu_smi())

    def test_has_npu_smi_not_found(self):
        """has_npu_smi returns False when npu-smi is not found.
        当找不到 npu-smi 时，has_npu_smi 返回 False。"""
        with patch.object(nvsmi_mod.shutil, "which", return_value=None):
            self.assertFalse(has_npu_smi())

    def test_has_xpu_smi_found(self):
        """has_xpu_smi returns True when xpu-smi is found.
        当找到 xpu-smi 时，has_xpu_smi 返回 True。"""
        with patch.object(
            nvsmi_mod.shutil, "which", return_value="/usr/bin/xpu-smi"
        ):
            self.assertTrue(has_xpu_smi())

    def test_has_xpu_smi_not_found(self):
        """has_xpu_smi returns False when xpu-smi is not found.
        当找不到 xpu-smi 时，has_xpu_smi 返回 False。"""
        with patch.object(nvsmi_mod.shutil, "which", return_value=None):
            self.assertFalse(has_xpu_smi())


@_SMI_SKIP
class TestGetGpuInfo(unittest.TestCase):
    """Test get_gpu_info function.
    测试 get_gpu_info 函数。"""

    def test_get_gpu_info_none_index(self):
        """get_gpu_info with None index.
        索引为 None 的 get_gpu_info。"""
        with patch.object(nvsmi_mod, "query_smi", return_value=[]):
            result = get_gpu_info(index=None)
            self.assertEqual(result, [])

    def test_get_gpu_info_int_index(self):
        """get_gpu_info with int index converts to list.
        整数索引的 get_gpu_info 会转换为列表。"""
        with patch.object(
            nvsmi_mod, "query_smi", return_value=[]
        ) as mock_query:
            result = get_gpu_info(index=0)
            mock_query.assert_called_once()
            # Index should be converted to ["0"]
            # 索引应转换为 ["0"]
            call_kwargs = mock_query.call_args[1]
            self.assertEqual(call_kwargs["index"], ["0"])


@_SMI_SKIP
class TestGetGpuUtil(unittest.TestCase):
    """Test get_gpu_util function.
    测试 get_gpu_util 函数。"""

    def test_get_gpu_util_default(self):
        """get_gpu_util uses nvidia query by default.
        get_gpu_util 默认使用 nvidia 查询。"""
        with (
            patch.object(nvsmi_mod, "query_smi", return_value=[]),
            patch("paddle.device.is_compiled_with_rocm", return_value=False),
            patch(
                "paddle.device.is_compiled_with_custom_device",
                return_value=False,
            ),
            patch("paddle.is_compiled_with_xpu", return_value=False),
        ):
            result = get_gpu_util()
            self.assertEqual(result, [])

    def test_get_gpu_util_rocm(self):
        """get_gpu_util uses rocm query when compiled with rocm.
        当使用 rocm 编译时，get_gpu_util 使用 rocm 查询。"""
        with (
            patch.object(nvsmi_mod, "query_rocm_smi", return_value=[]),
            patch("paddle.device.is_compiled_with_rocm", return_value=True),
        ):
            result = get_gpu_util()
            self.assertEqual(result, [])

    def test_get_gpu_util_npu(self):
        """get_gpu_util uses npu query for npu device.
        get_gpu_util 对 npu 设备使用 npu 查询。"""
        with (
            patch.object(nvsmi_mod, "query_npu_smi", return_value=[]),
            patch("paddle.device.is_compiled_with_rocm", return_value=False),
            patch(
                "paddle.device.is_compiled_with_custom_device",
                return_value=True,
            ),
            patch("paddle.is_compiled_with_xpu", return_value=False),
        ):
            result = get_gpu_util()
            self.assertEqual(result, [])

    def test_get_gpu_util_xpu(self):
        """get_gpu_util uses xpu query for xpu device.
        get_gpu_util 对 xpu 设备使用 xpu 查询。"""
        with (
            patch.object(nvsmi_mod, "query_xpu_smi", return_value=[]),
            patch("paddle.device.is_compiled_with_rocm", return_value=False),
            patch(
                "paddle.device.is_compiled_with_custom_device",
                return_value=False,
            ),
            patch("paddle.is_compiled_with_xpu", return_value=True),
        ):
            result = get_gpu_util()
            self.assertEqual(result, [])


@_SMI_SKIP
class TestGetGpuProcess(unittest.TestCase):
    """Test get_gpu_process function.
    测试 get_gpu_process 函数。"""

    def test_get_gpu_process_none_index(self):
        """get_gpu_process with None index.
        索引为 None 的 get_gpu_process。"""
        with patch.object(
            nvsmi_mod, "query_smi", return_value=[]
        ) as mock_query:
            result = get_gpu_process(index=None)
            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            self.assertEqual(call_kwargs["query_type"], "compute")

    def test_get_gpu_process_int_index(self):
        """get_gpu_process with int index.
        整数索引的 get_gpu_process。"""
        with patch.object(
            nvsmi_mod, "query_smi", return_value=[]
        ) as mock_query:
            result = get_gpu_process(index=1)
            call_kwargs = mock_query.call_args[1]
            self.assertEqual(call_kwargs["index"], ["1"])


if __name__ == "__main__":
    unittest.main()
