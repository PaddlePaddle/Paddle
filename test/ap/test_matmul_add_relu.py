# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import subprocess
import unittest

import numpy as np

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle/ap"


def GetPirProgram(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)


def IsCertainDevices():
    try:
        sp = subprocess.Popen(
            ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out_str = sp.communicate()[0].decode('utf-8')
        if 'A100' in out_str:
            return True
        else:
            return False
    except Exception as e:
        return False


class TestMatmulEpilogue(unittest.TestCase):
    def setUp(self):
        dtype = 'float16'
        x_shape = [32, 16, 16]
        self.x = paddle.randn(x_shape, dtype=dtype)
        self.x.stop_gradient = False

        y_shape = [16, 16]
        self.y = paddle.randn(y_shape, dtype=dtype)
        self.y.stop_gradient = False

        b_shape = [32, 16, 16]
        self.b = paddle.randn(b_shape, dtype=dtype)
        self.b.stop_gradient = False

    def getSubGraph(self):
        B = pct.DimVar(32)
        M = pct.DimVar(16)
        K = pct.DimVar(16)
        N = pct.DimVar(16)
        DType = pct.DTypeVar("T", "float16")

        def foo(
            x: pct.Tensor([B, M, K], DType),
            w: pct.Tensor([K, N], DType),
            b: pct.Tensor([B, M, N], DType),
        ):

            y = paddle.matmul(x, w)
            tmp = paddle.nn.functional.relu(y)
            tmp2 = tmp + b
            return tmp2

        return foo

    def test_subgraph(self):
        foo = self.getSubGraph()
        fused_foo = pcc.compile(
            foo, ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass"
        )
        generated_pir_program = GetPirProgram(
            fused_foo, [self.x, self.y, self.b]
        )
        self.assertTrue(
            'pd_op.ap_variadic' in generated_pir_program, "fusion failed"
        )
        if IsCertainDevices():
            ap_outs = fused_foo(self.x, self.y, self.b)
            dy_outs = foo(self.x, self.y, self.b)
            for dy_out, ap_out in zip(dy_outs, ap_outs):
                np.testing.assert_allclose(dy_out, ap_out, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
