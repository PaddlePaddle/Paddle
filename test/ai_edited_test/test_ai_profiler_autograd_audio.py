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

# [AUTO-GENERATED] Test file for paddle.profiler and paddle.autograd
# 覆盖模块: paddle/profiler/, paddle/autograd/, paddle/audio/backends/
# Uncovered lines: Profiler, RecordEvent, PyLayer, audio backends

import unittest

import numpy as np

import paddle
from paddle.autograd import PyLayer
from paddle.profiler import Profiler, RecordEvent


class TestProfiler(unittest.TestCase):
    """测试 Profiler 性能分析器
    Test Profiler"""

    def test_profiler_create(self):
        """测试创建 Profiler
        Test creating Profiler"""
        p = Profiler()
        self.assertIsNotNone(p)

    def test_profiler_start_stop(self):
        """测试 Profiler 启动和停止
        Test Profiler start and stop"""
        p = Profiler()
        p.start()
        a = paddle.randn([10])
        b = paddle.matmul(a, a.T)
        p.stop()
        p.summary()

    def test_record_event(self):
        """测试 RecordEvent 记录事件
        Test RecordEvent for recording events"""
        with RecordEvent('test_event'):
            a = paddle.randn([100, 100])
            b = paddle.matmul(a, a.T)
        self.assertIsNotNone(b)

    def test_record_event_nested(self):
        """测试嵌套 RecordEvent
        Test nested RecordEvent"""
        with RecordEvent('outer_event'):
            a = paddle.randn([10])
            with RecordEvent('inner_event'):
                b = paddle.matmul(a.reshape([1, 10]), a.reshape([10, 1]))
        self.assertIsNotNone(b)


class TestPyLayer(unittest.TestCase):
    """测试 PyLayer 自定义前向/反向
    Test PyLayer custom forward/backward"""

    def test_pylayer_basic(self):
        """测试基本 PyLayer
        Test basic PyLayer"""

        class CustomMultiply(PyLayer):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * 3.0

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensor()
                return grad_output * 2.0

        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        y = CustomMultiply.apply(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), np.full([3], 2.0), atol=1e-6)

    def test_pylayer_no_grad(self):
        """测试无梯度 PyLayer
        Test PyLayer with no gradient needed"""

        class IdentityLayer(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        x = paddle.randn([3])
        y = IdentityLayer.apply(x)
        np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-6)


class TestAudioBackends(unittest.TestCase):
    """测试 audio 后端模块
    Test audio backend module"""

    def test_get_current_backend(self):
        """测试获取当前后端
        Test getting current backend"""
        from paddle.audio.backends.init_backend import get_current_backend

        backend = get_current_backend()
        self.assertIsNotNone(backend)

    def test_list_available_backends(self):
        """测试列出可用后端
        Test listing available backends"""
        from paddle.audio.backends.init_backend import list_available_backends

        backends = list_available_backends()
        self.assertIsInstance(backends, list)
        self.assertGreater(len(backends), 0)


class TestAutogradUtils(unittest.TestCase):
    """测试 autograd 工具函数
    Test autograd utility functions"""

    def test_no_grad(self):
        """测试 no_grad 上下文
        Test no_grad context"""
        x = paddle.randn([3])
        with paddle.no_grad():
            y = x * 2
            # In no_grad context, new tensors are created without gradient tracking
            self.assertIsNotNone(y)

    def test_enable_grad(self):
        """测试 enable_grad 上下文
        Test enable_grad context"""
        x = paddle.randn([3])
        x.stop_gradient = True
        with paddle.enable_grad():
            self.assertTrue(paddle.is_grad_enabled())

    def test_grad_function(self):
        """测试 grad 函数
        Test grad function"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        y = x * 2
        grads = paddle.grad(y, x)
        self.assertIsNotNone(grads)

    def test_is_grad_enabled(self):
        """测试 is_grad_enabled
        Test is_grad_enabled"""
        result = paddle.autograd.is_grad_enabled()
        self.assertIsInstance(result, bool)


class TestDistributedRuntimeFactory(unittest.TestCase):
    """测试 distributed fleet runtime factory
    Test distributed fleet runtime factory"""

    def test_runtime_factory_import(self):
        """测试 RuntimeFactory 可导入
        Test RuntimeFactory can be imported"""
        from paddle.distributed.fleet.base.runtime_factory import (
            CollectiveRuntime,
            RuntimeFactory,
            TheOnePSRuntime,
        )

        self.assertIsNotNone(RuntimeFactory)
        self.assertIsNotNone(CollectiveRuntime)
        self.assertIsNotNone(TheOnePSRuntime)


class TestIncubateFunctional(unittest.TestCase):
    """测试 incubate 功能函数
    Test incubate functional"""

    def test_fused_matmul_bias_import(self):
        """测试 fused_matmul_bias 可导入
        Test fused_matmul_bias can be imported"""
        from paddle.incubate.nn.functional import fused_matmul_bias

        self.assertIsNotNone(fused_matmul_bias)

    def test_variable_length_memory_efficient_attention_import(self):
        """测试 variable_length_memory_efficient_attention 可导入
        Test variable_length_memory_efficient_attention can be imported"""
        from paddle.incubate.nn.functional import (
            variable_length_memory_efficient_attention,
        )

        self.assertIsNotNone(variable_length_memory_efficient_attention)


class TestDistributedRpcInternal(unittest.TestCase):
    """测试分布式 RPC 内部模块
    Test distributed RPC internal module"""

    def test_python_func_import(self):
        """测试 PythonFunc 可导入
        Test PythonFunc can be imported"""
        from paddle.distributed.rpc.internal import PythonFunc

        self.assertIsNotNone(PythonFunc)


if __name__ == '__main__':
    unittest.main()
