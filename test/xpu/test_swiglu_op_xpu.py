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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import swiglu as fused_swiglu_impl


def swiglu(x, y, out_grad):
    if isinstance(x, np.ndarray):
        x = paddle.to_tensor(x)
        y = paddle.to_tensor(y)
        out_grad = paddle.to_tensor(out_grad)

    origin_x = x.detach().clone()
    origin_x.stop_gradient = False
    x = origin_x

    origin_y = y.detach().clone()
    origin_y.stop_gradient = False
    y = origin_y

    dtype = x.dtype
    need_convert = False
    assert dtype == y.dtype
    output_dtype = dtype

    out = F.silu(x) * y
    if need_convert:
        out = out.astype(dtype)
    out.backward(out_grad)
    ret = [
        out.astype(output_dtype),
        origin_x.grad.astype(output_dtype),
        origin_y.grad.astype(output_dtype),
    ]
    return ret


def fused_swiglu(x, y, out_grad):
    x = x.detach().clone()
    x.stop_gradient = False
    if y is not None:
        y = y.detach().clone()
        y.stop_gradient = False
    out = fused_swiglu_impl(x, y)
    out.backward(out_grad)

    output_dtype = x.dtype
    ret = [
        out.astype(output_dtype),
    ]
    if y is not None:
        x_grad, y_grad = x.grad, y.grad
    else:
        x_grad, y_grad = paddle.split(x.grad, 2, axis=-1)

    ret.append(x_grad.astype(output_dtype))
    ret.append(y_grad.astype(output_dtype))
    return ret


tol_map = {
    paddle.float64: [1e-8, 1e-8],
    paddle.float32: [1e-6, 1e-6],
    paddle.float16: [1e-3, 1e-3],
    paddle.bfloat16: [1e-2, 1e-2],
}


class TestSwiGLUDygraph(unittest.TestCase):
    def setUp(self):
        self.init_case()
        self.seed = 1234

    def init_case(self):
        self.shape = []
        self.shape.append([8, 100])
        self.shape.append([4, 102])

    def check_dygraph_impl(self, device, shape, dtype):
        x = paddle.randn(shape, dtype=dtype)
        y = paddle.randn(shape, dtype=dtype)
        out_grad = paddle.randn(shape, dtype=dtype)

        ret1 = swiglu(x, y, out_grad)
        ret2 = fused_swiglu(x, y, out_grad)
        ret3 = fused_swiglu(paddle.concat([x, y], axis=-1), None, out_grad)

        atol, rtol = tol_map[dtype]
        err_msg = (
            f"Failed when device = {device}, dtype = {dtype}, shape = {shape}"
        )
        for t1, t2, t3 in zip(ret1, ret2, ret3):
            t1, t2, t3 = t1.numpy(), t2.numpy(), t3.numpy()
            np.testing.assert_allclose(
                t1, t2, atol=atol, rtol=rtol, err_msg=err_msg
            )
            np.testing.assert_equal(t2, t3, err_msg=err_msg)

    def check_dygraph(self, shape):
        metas = []
        metas.append(('xpu', paddle.float32))
        metas.append(('xpu', paddle.float64))
        # Enable in KL3
        # metas.append(('xpu', paddle.float16))
        # metas.append(('xpu', paddle.bfloat16))

        for device, dtype in metas:
            origin_device = paddle.get_device()
            paddle.set_device(device)
            for with_split in [True]:
                self.check_dygraph_impl(device, shape, dtype)
            paddle.set_device(origin_device)

    def check_static_graph(self, shape, dtype="float32"):
        x = paddle.static.data(name='x', shape=shape, dtype=dtype)
        y = paddle.static.data(name='y', shape=shape, dtype=dtype)
        concated_x = paddle.static.data(
            name='concated_x',
            shape=[*shape[:-1], shape[-1] * 2],
            dtype=dtype,
        )
        out1 = fused_swiglu_impl(x, y)
        out2 = fused_swiglu_impl(concated_x)

        concated_x_np = np.random.random(concated_x.shape).astype(dtype)
        x_np, y_np = np.split(concated_x_np, 2, axis=-1)

        exe = paddle.static.Executor()
        t1, t2 = exe.run(
            feed={'x': x_np, 'y': y_np, 'concated_x': concated_x_np},
            fetch_list=[out1, out2],
        )
        np.testing.assert_equal(t1, t2)

    def check_main(self, shape):
        self.check_dygraph(shape)
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            self.check_static_graph(shape)
        paddle.disable_static()

    def test_main(self):
        for i in self.shape:
            self.check_main(i)


class TestSwigluOp(TestSwiGLUDygraph):
    def init_case(self):
        self.shape = [[1, 4096, 1376], [1, 4096, 11008]]


if __name__ == "__main__":
    unittest.main()
