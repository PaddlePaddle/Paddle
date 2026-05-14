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

# [AUTO-GENERATED] Do not edit manually.
# Target source: paddle/phi/kernels/cpu/send_ue_recv_kernel.cc
# Generated for exercising C++ CPU kernel: SendUERecvKernel
#
# 测试图消息传递 SendUERecv CPU 内核
# Tests for Graph message passing SendUERecv CPU kernel

import unittest

import numpy as np

import paddle
import paddle.geometric as pg


class TestSendUERecvBasic(unittest.TestCase):
    """基本 SendUERecv 测试 / Basic SendUERecv tests"""

    def setUp(self):
        """初始化测试环境 / Setup test environment"""
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        """清理测试环境 / Teardown test environment"""
        paddle.enable_static()

    def test_send_ue_recv_add_sum(self):
        """测试 ADD+SUM 消息传递
        Test ADD message with SUM reduce
        """
        x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
        y = paddle.to_tensor([1, 1, 1, 1], dtype="float32")
        src_index = paddle.to_tensor([0, 1, 2, 0], dtype="int64")
        dst_index = paddle.to_tensor([1, 2, 1, 0], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "sum")

        # node 0: receives from src 0 with y=1 (last edge), x[0]+1 = [1,3,4]
        # node 1: receives from src 0 (y=1) and src 2 (y=1): x[0]+1 + x[2]+1 = [3,9,11]
        # node 2: receives from src 1 (y=1): x[1]+1 = [2,5,6]
        self.assertEqual(out.shape, [3, 3])
        expected_0 = np.array([1, 3, 4], dtype="float32")
        np.testing.assert_allclose(out[0].numpy(), expected_0, atol=1e-5)

    def test_send_ue_recv_add_mean(self):
        """测试 ADD+MEAN 消息传递
        Test ADD message with MEAN reduce
        """
        x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
        y = paddle.to_tensor([1, 1, 1, 1], dtype="float32")
        src_index = paddle.to_tensor([0, 1, 2, 0], dtype="int64")
        dst_index = paddle.to_tensor([1, 2, 1, 0], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "mean")

        self.assertEqual(out.shape, [3, 3])
        # node 0: receives 1 msg (from src 0): x[0]+1 = [1,3,4]
        # node 1: receives 2 msgs, mean = (x[0]+1 + x[2]+1)/2 = ([1,3,4]+[3,7,8])/2 = [2,5,6]
        # node 2: receives 1 msg (from src 1): x[1]+1 = [2,5,6]
        expected_1 = np.array([2.0, 5.0, 6.0], dtype="float32")
        np.testing.assert_allclose(out[1].numpy(), expected_1, atol=1e-5)

    def test_send_ue_recv_mul_sum(self):
        """测试 MUL+SUM 消息传递
        Test MUL message with SUM reduce
        """
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
        y = paddle.to_tensor([2, 2, 2], dtype="float32")
        src_index = paddle.to_tensor([0, 1, 0], dtype="int64")
        dst_index = paddle.to_tensor([1, 2, 1], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "mul", "sum")

        # node 1: receives from src 0 twice: x[0]*2 + x[0]*2 = [4,8,12]
        self.assertEqual(out.shape, [3, 3])
        expected_1 = np.array([4, 8, 12], dtype="float32")
        np.testing.assert_allclose(out[1].numpy(), expected_1, atol=1e-5)

    def test_send_ue_recv_add_max(self):
        """测试 ADD+MAX 消息传递
        Test ADD message with MAX reduce
        """
        x = paddle.to_tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype="float32")
        y = paddle.to_tensor([1, 1], dtype="float32")
        src_index = paddle.to_tensor([0, 1], dtype="int64")
        dst_index = paddle.to_tensor([2, 2], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "max")

        # node 2: receives x[0]+1=[2,2,2] and x[1]+1=[3,3,3], max = [3,3,3]
        self.assertEqual(out.shape, [3, 3])
        expected_2 = np.array([3, 3, 3], dtype="float32")
        np.testing.assert_allclose(out[2].numpy(), expected_2, atol=1e-5)

    def test_send_ue_recv_add_min(self):
        """测试 ADD+MIN 消息传递
        Test ADD message with MIN reduce
        """
        x = paddle.to_tensor([[5, 5, 5], [1, 1, 1], [3, 3, 3]], dtype="float32")
        y = paddle.to_tensor([1, 1], dtype="float32")
        src_index = paddle.to_tensor([0, 1], dtype="int64")
        dst_index = paddle.to_tensor([2, 2], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "min")

        # node 2: receives x[0]+1=[6,6,6] and x[1]+1=[2,2,2], min = [2,2,2]
        expected_2 = np.array([2, 2, 2], dtype="float32")
        np.testing.assert_allclose(out[2].numpy(), expected_2, atol=1e-5)


class TestSendUERecvWithOutSize(unittest.TestCase):
    """带 out_size 参数的 SendUERecv 测试 / SendUERecv tests with out_size"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_send_ue_recv_out_size(self):
        """测试指定 out_size 的消息传递
        Test message passing with specified out_size
        """
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
        y = paddle.to_tensor([1], dtype="float32")
        src_index = paddle.to_tensor([0], dtype="int64")
        dst_index = paddle.to_tensor([3], dtype="int64")

        out = pg.send_ue_recv(
            x, y, src_index, dst_index, "add", "sum", out_size=4
        )

        self.assertEqual(out.shape, [4, 2])
        # Node 3 should have x[0]+1 = [2, 3]
        np.testing.assert_allclose(out[3].numpy(), [2, 3], atol=1e-5)
        # Other nodes should be zero (no messages received)
        np.testing.assert_allclose(out[0].numpy(), [0, 0], atol=1e-5)


class TestSendUERecvEdgeCases(unittest.TestCase):
    """SendUERecv 边界情况测试 / SendUERecv edge case tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_send_ue_recv_single_message(self):
        """测试单条消息传递
        Test single message passing
        """
        x = paddle.to_tensor([[10, 20]], dtype="float32")
        y = paddle.to_tensor([5], dtype="float32")
        src_index = paddle.to_tensor([0], dtype="int64")
        dst_index = paddle.to_tensor([0], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "sum")

        # node 0: x[0]+5 = [15, 25]
        np.testing.assert_allclose(out[0].numpy(), [15, 25], atol=1e-5)

    def test_send_ue_recv_empty_indices(self):
        """测试空索引（无消息）
        Test with empty indices (no messages)
        """
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
        y = paddle.to_tensor([], dtype="float32")
        src_index = paddle.to_tensor([], dtype="int64")
        dst_index = paddle.to_tensor([], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "sum")

        # No messages, output should be all zeros
        np.testing.assert_allclose(
            out.numpy(), np.zeros((2, 2), dtype="float32"), atol=1e-5
        )

    def test_send_ue_recv_int32_index(self):
        """测试 int32 类型索引
        Test with int32 index type
        """
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
        y = paddle.to_tensor([1], dtype="float32")
        src_index = paddle.to_tensor([0], dtype="int32")
        dst_index = paddle.to_tensor([1], dtype="int32")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "sum")

        np.testing.assert_allclose(out[1].numpy(), [2, 3], atol=1e-5)

    def test_send_ue_recv_float64(self):
        """测试 float64 类型的消息传递
        Test message passing with float64
        """
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype="float64")
        y = paddle.to_tensor([1, 1], dtype="float64")
        src_index = paddle.to_tensor([0, 1], dtype="int64")
        dst_index = paddle.to_tensor([1, 0], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "sum")

        self.assertEqual(out.dtype, paddle.float64)
        # node 0: receives from src 1: x[1]+1 = [4,5]
        # node 1: receives from src 0: x[0]+1 = [2,3]
        np.testing.assert_allclose(out[0].numpy(), [4, 5], atol=1e-8)
        np.testing.assert_allclose(out[1].numpy(), [2, 3], atol=1e-8)

    def test_send_ue_recv_self_loop(self):
        """测试自环消息传递
        Test self-loop message passing
        """
        x = paddle.to_tensor([[1, 1]], dtype="float32")
        y = paddle.to_tensor([10], dtype="float32")
        src_index = paddle.to_tensor([0], dtype="int64")
        dst_index = paddle.to_tensor([0], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "sum")

        np.testing.assert_allclose(out[0].numpy(), [11, 11], atol=1e-5)

    def test_send_ue_recv_multiple_edges_same_dst(self):
        """测试多条边指向同一目标节点
        Test multiple edges to same destination
        """
        x = paddle.to_tensor([[1, 0], [0, 1], [0, 0]], dtype="float32")
        y = paddle.to_tensor([1, 1, 1], dtype="float32")
        src_index = paddle.to_tensor([0, 1, 0], dtype="int64")
        dst_index = paddle.to_tensor([2, 2, 2], dtype="int64")

        out = pg.send_ue_recv(x, y, src_index, dst_index, "add", "sum")

        # node 2: (x[0]+1) + (x[1]+1) + (x[0]+1) = [2,1] + [1,2] + [2,1] = [5,4]
        np.testing.assert_allclose(out[2].numpy(), [5, 4], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
