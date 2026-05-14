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

# [AUTO-GENERATED] Unit test for paddle.distributed.launch.utils.kv_client
# 自动生成的单测，覆盖 kv_client 模块中未覆盖的代码
# Target: cover uncovered lines 21-36, 38-49, 51-59, 61-71, 73-78, 80-88
#   in python/paddle/distributed/launch/utils/kv_client.py
# 未覆盖行: KVClient.__init__ endpoint格式化, put/get/get_prefix/delete/wait_server_ready

import sys
import unittest
from unittest.mock import MagicMock, patch

# Import through the working path, then get module from sys.modules for patching
# 通过可用的路径导入，然后从 sys.modules 获取模块用于 patching

kv_mod = sys.modules['paddle.distributed.launch.utils.kv_client']
from paddle.distributed.launch.utils.kv_client import KVClient


class TestKVClientInit(unittest.TestCase):
    """Test KVClient initialization.
    测试 KVClient 初始化。"""

    def test_init_with_http_prefix(self):
        """KVClient with http:// prefix keeps endpoint as-is.
        带有 http:// 前缀的 KVClient 保持端点不变。"""
        client = KVClient("http://localhost:2379")
        self.assertEqual(client.endpoint, "http://localhost:2379")

    def test_init_without_http_prefix(self):
        """KVClient without http:// prefix adds http://.
        不带 http:// 前缀的 KVClient 会添加 http://。"""
        client = KVClient("localhost:2379")
        self.assertEqual(client.endpoint, "http://localhost:2379")

    def test_init_default_endpoint(self):
        """KVClient default endpoint is http://localhost:2379.
        KVClient 默认端点为 http://localhost:2379。"""
        client = KVClient()
        self.assertEqual(client.endpoint, "http://localhost:2379")

    def test_init_custom_endpoint(self):
        """KVClient with custom endpoint.
        使用自定义端点的 KVClient。"""
        client = KVClient("http://custom:8080")
        self.assertEqual(client.endpoint, "http://custom:8080")


class TestKVClientPut(unittest.TestCase):
    """Test KVClient.put method.
    测试 KVClient.put 方法。"""

    def setUp(self):
        self.client = KVClient("http://localhost:2379")

    def test_put_with_slash_key(self):
        """put with key already starting with /.
        以 / 开头的键调用 put。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(kv_mod.httpx, "post", return_value=mock_response):
            result = self.client.put("/my/key", "value")
            self.assertTrue(result)
            kv_mod.httpx.post.assert_called_once_with(
                "http://localhost:2379/my/key",
                data="value",
                timeout=None,
                follow_redirects=True,
            )

    def test_put_without_slash_key(self):
        """put with key not starting with / adds prefix.
        不以 / 开头的键调用 put 会添加前缀。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(kv_mod.httpx, "post", return_value=mock_response):
            result = self.client.put("my/key", "value")
            self.assertTrue(result)
            kv_mod.httpx.post.assert_called_once_with(
                "http://localhost:2379/my/key",
                data="value",
                timeout=None,
                follow_redirects=True,
            )

    def test_put_non_200_status(self):
        """put returns False when status code is not 200.
        状态码不是200时 put 返回 False。"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch.object(kv_mod.httpx, "post", return_value=mock_response):
            result = self.client.put("/key", "value")
            self.assertFalse(result)

    def test_put_exception(self):
        """put returns False on exception.
        发生异常时 put 返回 False。"""
        with patch.object(
            kv_mod.httpx,
            "post",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = self.client.put("/key", "value")
            self.assertFalse(result)


class TestKVClientGet(unittest.TestCase):
    """Test KVClient.get method.
    测试 KVClient.get 方法。"""

    def setUp(self):
        self.client = KVClient("http://localhost:2379")

    def test_get_with_slash_key(self):
        """get with key starting with / returns value.
        以 / 开头的键调用 get 返回值。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"/my/key": "hello"}
        with patch.object(kv_mod.httpx, "get", return_value=mock_response):
            result = self.client.get("/my/key")
            self.assertEqual(result, "hello")
            kv_mod.httpx.get.assert_called_once_with(
                "http://localhost:2379/my/key",
                timeout=None,
                follow_redirects=True,
            )

    def test_get_without_slash_key(self):
        """get with key not starting with / adds prefix.
        不以 / 开头的键调用 get 会添加前缀。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"/key": "world"}
        with patch.object(kv_mod.httpx, "get", return_value=mock_response):
            result = self.client.get("key")
            self.assertEqual(result, "world")

    def test_get_missing_key_in_response(self):
        """get returns empty string when key not in response json.
        当响应 json 中没有该键时，get 返回空字符串。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"/other/key": "value"}
        with patch.object(kv_mod.httpx, "get", return_value=mock_response):
            result = self.client.get("/missing/key")
            self.assertEqual(result, "")

    def test_get_non_200_status(self):
        """get returns 'error' when status code is not 200.
        状态码不是200时 get 返回 'error'。"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch.object(kv_mod.httpx, "get", return_value=mock_response):
            result = self.client.get("/key")
            self.assertEqual(result, "error")

    def test_get_exception(self):
        """get returns empty string on exception.
        发生异常时 get 返回空字符串。"""
        with patch.object(
            kv_mod.httpx,
            "get",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = self.client.get("/key")
            self.assertEqual(result, "")


class TestKVClientGetPrefix(unittest.TestCase):
    """Test KVClient.get_prefix method.
    测试 KVClient.get_prefix 方法。"""

    def setUp(self):
        self.client = KVClient("http://localhost:2379")

    def test_get_prefix_success(self):
        """get_prefix returns JSON dict on success.
        成功时 get_prefix 返回 JSON 字典。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "/workers/1": "rank1",
            "/workers/2": "rank2",
        }
        with patch.object(kv_mod.httpx, "get", return_value=mock_response):
            result = self.client.get_prefix("/workers")
            self.assertEqual(
                result, {"/workers/1": "rank1", "/workers/2": "rank2"}
            )

    def test_get_prefix_without_slash(self):
        """get_prefix with key not starting with / adds prefix.
        不以 / 开头的键调用 get_prefix 会添加前缀。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"/workers/1": "rank1"}
        with patch.object(kv_mod.httpx, "get", return_value=mock_response):
            result = self.client.get_prefix("workers")
            self.assertEqual(result, {"/workers/1": "rank1"})

    def test_get_prefix_non_200(self):
        """get_prefix returns None when status is not 200.
        状态码不是200时 get_prefix 返回 None。"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch.object(kv_mod.httpx, "get", return_value=mock_response):
            result = self.client.get_prefix("/workers")
            self.assertIsNone(result)

    def test_get_prefix_exception(self):
        """get_prefix returns empty string on exception.
        发生异常时 get_prefix 返回空字符串。"""
        with patch.object(
            kv_mod.httpx,
            "get",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = self.client.get_prefix("/workers")
            self.assertEqual(result, "")


class TestKVClientDelete(unittest.TestCase):
    """Test KVClient.delete method.
    测试 KVClient.delete 方法。"""

    def setUp(self):
        self.client = KVClient("http://localhost:2379")

    def test_delete_success(self):
        """delete returns True on success.
        成功时 delete 返回 True。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(kv_mod.httpx, "delete", return_value=mock_response):
            result = self.client.delete("/key")
            self.assertTrue(result)
            kv_mod.httpx.delete.assert_called_once_with(
                "http://localhost:2379/key",
                timeout=None,
                follow_redirects=True,
            )

    def test_delete_without_slash(self):
        """delete with key not starting with / adds prefix.
        不以 / 开头的键调用 delete 会添加前缀。"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(kv_mod.httpx, "delete", return_value=mock_response):
            result = self.client.delete("key")
            self.assertTrue(result)

    def test_delete_non_200(self):
        """delete returns False when status is not 200.
        状态码不是200时 delete 返回 False。"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch.object(kv_mod.httpx, "delete", return_value=mock_response):
            result = self.client.delete("/key")
            self.assertFalse(result)

    def test_delete_exception(self):
        """delete returns False on exception.
        发生异常时 delete 返回 False。"""
        with patch.object(
            kv_mod.httpx,
            "delete",
            side_effect=ConnectionError("Connection refused"),
        ):
            result = self.client.delete("/key")
            self.assertFalse(result)


class TestKVClientWaitServerReady(unittest.TestCase):
    """Test KVClient.wait_server_ready method.
    测试 KVClient.wait_server_ready 方法。"""

    def test_wait_server_ready_immediate_success(self):
        """wait_server_ready returns True immediately when healthy.
        当服务器健康时 wait_server_ready 立即返回 True。"""
        client = KVClient("http://localhost:2379")
        # time.side_effect=[0, 1]: first call sets end=0+3=3, second call 1<3 enters loop
        # time.side_effect=[0, 1]: 第一次调用设置 end=0+3=3，第二次调用 1<3 进入循环
        with (
            patch.object(kv_mod.time, "time", side_effect=[0, 1, 10]),
            patch.object(client, "get", return_value="ok"),
        ):
            result = client.wait_server_ready(timeout=3)
            self.assertTrue(result)

    def test_wait_server_ready_timeout(self):
        """wait_server_ready returns None when timeout is reached.
        超时时 wait_server_ready 返回 None。"""
        client = KVClient("http://localhost:2379")
        with (
            patch.object(kv_mod.time, "time", side_effect=[0, 1, 2, 3, 4]),
            patch.object(client, "get", return_value=""),
        ):
            result = client.wait_server_ready(timeout=3)
            self.assertIsNone(result)

    def test_wait_server_ready_eventual_success(self):
        """wait_server_ready returns True after retries.
        重试后 wait_server_ready 返回 True。"""
        client = KVClient("http://localhost:2379")
        # side_effect=[0, 1, 2, 3, 10]: end=3, loop at t=1 (get returns ''), t=2 (get returns ''), t=3 (>=end, exit)
        # But need t < end to enter loop, so t=1 (end=3): enter, get=''; t=2: enter, get=''; t=3: 3<3 false, exit
        # Need: end = 0 + 3 = 3, then time.time() at 1, 2, 3
        # Actually 3 < 3 is False, so only 2 iterations. Adjust to have 3 iterations.
        with (
            patch.object(kv_mod.time, "time", side_effect=[0, 1, 2, 3, 10]),
            patch.object(client, "get", side_effect=["", "", "ok"]),
        ):
            result = client.wait_server_ready(timeout=5)
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
