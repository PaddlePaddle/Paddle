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

import os
import unittest

import numpy as np

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
)

os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle_ap_workspace"


def GetPirProgram(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)


def IsSupportDevice():
    if paddle.is_compiled_with_cuda():
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        return cc == 80

    if paddle.is_compiled_with_rocm():
        return False

    return False


class TestAttentionEpilogue(unittest.TestCase):
    """`memory_efficient_attention + (add bias) + relu` elementwise epilogue."""

    def setUp(self):
        self.origin_flags = paddle.get_flags(['FLAGS_deny_cinn_ops'])
        paddle.set_flags({'FLAGS_deny_cinn_ops': ""})

        dtype = 'float16'

        # BMHK layout: [batch, seqlen, num_heads, head_dim].
        self.B, self.M, self.N, self.H, self.D = 2, 128, 128, 8, 64
        self.q = paddle.randn([self.B, self.M, self.H, self.D], dtype=dtype)
        self.k = paddle.randn([self.B, self.N, self.H, self.D], dtype=dtype)
        self.v = paddle.randn([self.B, self.N, self.H, self.D], dtype=dtype)
        # Per-head bias broadcast along batch / query, same shape as output.
        self.bias = paddle.randn([self.B, self.M, self.H, self.D], dtype=dtype)
        for tensor in (self.q, self.k, self.v, self.bias):
            tensor.stop_gradient = False

    def tearDown(self):
        paddle.set_flags(self.origin_flags)

    def getSubGraph(self):
        B = pct.DimVar(self.B)
        M = pct.DimVar(self.M)
        N = pct.DimVar(self.N)
        H = pct.DimVar(self.H)
        D = pct.DimVar(self.D)
        DType = pct.DTypeVar("T", "float16")

        def foo(
            q: pct.Tensor([B, M, H, D], DType),
            k: pct.Tensor([B, N, H, D], DType),
            v: pct.Tensor([B, N, H, D], DType),
            bias: pct.Tensor([B, M, H, D], DType),
        ):
            # training=False -> is_test, only `output` is used downstream.
            o = memory_efficient_attention(q, k, v, training=False)
            return paddle.nn.functional.relu(o + bias)

        return foo

    def test_subgraph(self):
        foo = self.getSubGraph()
        backend_device = 'dcu' if paddle.is_compiled_with_rocm() else 'cuda'
        fused_foo = pcc.compile(
            foo,
            ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass",
            backend_device=backend_device,
        )
        generated_pir_program = GetPirProgram(
            fused_foo, [self.q, self.k, self.v, self.bias]
        )
        if IsSupportDevice():
            self.assertIn(
                'pd_op.ap_variadic', generated_pir_program, "fusion failed"
            )
            ap_outs = fused_foo(self.q, self.k, self.v, self.bias)
            dy_outs = foo(self.q, self.k, self.v, self.bias)
            for dy_out, ap_out in zip(dy_outs, ap_outs):
                np.testing.assert_allclose(dy_out, ap_out, atol=1e-1)
        else:
            # The FMHA backend is compiled for sm_80 only, so any other device
            # (including ROCm, which has no attention backend at all) has to
            # keep the subgraph on the phi kernels.
            self.assertNotIn('pd_op.ap_variadic', generated_pir_program)


class TestAttentionEpilogueDynamicHeadDimNotFused(unittest.TestCase):
    """A symbolic head_dim must not be fused.

    The template bakes head_dim into the cutlass kernel template arguments
    (``kMaxK`` / ``Alignment``), so a non-constant head_dim would emit an
    undeclared ``D`` and fail to compile; ``_has_static_head_dim`` rejects it.
    """

    def setUp(self):
        self.origin_flags = paddle.get_flags(['FLAGS_deny_cinn_ops'])
        paddle.set_flags({'FLAGS_deny_cinn_ops': ""})

        dtype = 'float16'
        self.B, self.M, self.N, self.H, self.D = 2, 128, 128, 8, 64
        self.q = paddle.randn([self.B, self.M, self.H, self.D], dtype=dtype)
        self.k = paddle.randn([self.B, self.N, self.H, self.D], dtype=dtype)
        self.v = paddle.randn([self.B, self.N, self.H, self.D], dtype=dtype)
        self.bias = paddle.randn([self.B, self.M, self.H, self.D], dtype=dtype)
        for tensor in (self.q, self.k, self.v, self.bias):
            tensor.stop_gradient = False

    def tearDown(self):
        paddle.set_flags(self.origin_flags)

    def getSubGraph(self):
        B = pct.DimVar(self.B)
        M = pct.DimVar(self.M)
        N = pct.DimVar(self.N)
        H = pct.DimVar(self.H)
        D = pct.DimVar("D")  # symbolic head_dim
        DType = pct.DTypeVar("T", "float16")

        def foo(
            q: pct.Tensor([B, M, H, D], DType),
            k: pct.Tensor([B, N, H, D], DType),
            v: pct.Tensor([B, N, H, D], DType),
            bias: pct.Tensor([B, M, H, D], DType),
        ):
            o = memory_efficient_attention(q, k, v, training=False)
            return paddle.nn.functional.relu(o + bias)

        return foo

    def test_subgraph(self):
        foo = self.getSubGraph()
        backend_device = 'dcu' if paddle.is_compiled_with_rocm() else 'cuda'
        fused_foo = pcc.compile(
            foo,
            ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass",
            backend_device=backend_device,
        )
        generated_pir_program = GetPirProgram(
            fused_foo, [self.q, self.k, self.v, self.bias]
        )
        self.assertNotIn(
            'pd_op.ap_variadic',
            generated_pir_program,
            "a dynamic head_dim must not be fused",
        )


if __name__ == "__main__":
    unittest.main()
