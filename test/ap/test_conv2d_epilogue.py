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


class TestConv2dEpilogue(unittest.TestCase):
    def setUp(self):
        self.origin_flags = paddle.get_flags(
            ['FLAGS_manually_trans_conv_filter', 'FLAGS_deny_cinn_ops']
        )
        paddle.set_flags(
            {
                'FLAGS_manually_trans_conv_filter': True,
                'FLAGS_deny_cinn_ops': "transpose",
            }
        )

        dtype = 'float16'

        # The cutlass implicit gemm backend requires an NHWC activation.
        x_shape = [32, 8, 8, 16]
        self.x = paddle.randn(x_shape, dtype=dtype)
        self.x.stop_gradient = False

        # Native paddle (KCRS) filter, transposed to KRSC inside the model.
        w_shape = [16, 16, 3, 3]
        self.w = paddle.randn(w_shape, dtype=dtype)
        self.w.stop_gradient = False

    def tearDown(self):
        paddle.set_flags(self.origin_flags)

    def getSubGraph(self):
        N = pct.DimVar(32)
        H = pct.DimVar(8)
        W = pct.DimVar(8)
        C = pct.DimVar(16)
        O = pct.DimVar(16)
        KH = pct.DimVar(3)
        KW = pct.DimVar(3)
        DType = pct.DTypeVar("T", "float16")

        def foo(
            x: pct.Tensor([N, H, W, C], DType),
            w: pct.Tensor([O, C, KH, KW], DType),
        ):
            # KCRS -> KRSC
            w = paddle.transpose(w, [0, 2, 3, 1])
            y = paddle.nn.functional.conv2d(x, w, padding=1, data_format="NHWC")
            return paddle.nn.functional.relu(y)

        return foo

    def test_subgraph(self):
        foo = self.getSubGraph()
        backend_device = 'dcu' if paddle.is_compiled_with_rocm() else 'cuda'
        fused_foo = pcc.compile(
            foo,
            ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass",
            backend_device=backend_device,
        )
        generated_pir_program = GetPirProgram(fused_foo, [self.x, self.w])
        self.assertTrue(
            'pd_op.ap_variadic' in generated_pir_program, "fusion failed"
        )
        if IsSupportDevice():
            ap_outs = fused_foo(self.x, self.w)
            dy_outs = foo(self.x, self.w)
            for dy_out, ap_out in zip(dy_outs, ap_outs):
                np.testing.assert_allclose(dy_out, ap_out, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
