# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import torch

import paddle


def convert_dtype_to_torch_type(dtype):
    if dtype in ["float32", np.float32]:
        return torch.float32
    elif dtype in ['float16', np.float16]:
        return torch.float16
    elif dtype in ['bfloat16', np.uint16]:
        return torch.bfloat16


class TestLogSoftmax_FP32(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.rtol = 1e-6
        self.atol = 1e-6

    def gen_np_inputs(self):
        self.np_x = np.random.random(size=self.shape).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.shape).astype("float32") - 0.5

    def gen_paddle_inputs(self):
        p_x = paddle.to_tensor(self.np_x, dtype=self.dtype, place="gpu")
        p_x.stop_gradient = False
        p_dout = paddle.to_tensor(self.np_dout, dtype=self.dtype, place="gpu")
        p_dout.stop_gradient = False
        return p_x, p_dout

    def gen_paddle_static_inputs(self):
        p_x_static = paddle.static.data("x", self.shape, dtype="float32")
        p_x_static.stop_gradient = False
        p_dout_static = paddle.static.data("dout", self.shape, dtype="float32")
        p_dout_static.stop_gradient = False
        return p_x_static, p_dout_static

    def gen_torch_inputs(self):
        t_x = torch.tensor(
            self.np_x,
            device="cuda",
            dtype=convert_dtype_to_torch_type(self.dtype),
            requires_grad=True,
        )
        t_dout = torch.tensor(
            self.np_dout,
            device="cuda",
            dtype=convert_dtype_to_torch_type(self.dtype),
            requires_grad=True,
        )
        return t_x, t_dout

    def cal_paddle_res(self, x, dout):
        out = paddle.nn.functional.log_softmax(
            x, axis=self.axis, dtype=self.dtype
        )
        out_grads = paddle.grad(out, x, grad_outputs=dout)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, "float32")
            out_grads = [
                paddle.cast(out_grad, "float32") for out_grad in out_grads
            ]
        return out, out_grads

    def cal_paddle_static_res(self, x, dout):
        if self.dtype == "bfloat16" or self.dtype == "float16":
            x = paddle.cast(x, dtype=self.dtype)
            dout = paddle.cast(dout, dtype=self.dtype)
        out = paddle.nn.functional.log_softmax(x, axis=self.axis)
        out_grads = paddle.static.gradients(out, x, target_gradients=dout)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, "float32")
            out_grads = [
                paddle.cast(out_grad, "float32") for out_grad in out_grads
            ]
        return out, out_grads

    def cal_torch_res(self, x, dout):
        out = torch.nn.functional.log_softmax(
            x, dim=self.axis, dtype=convert_dtype_to_torch_type(self.dtype)
        )
        out_grads = torch.autograd.grad(out, x, grad_outputs=dout)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = [
                out_grad.to(dtype=torch.float32) for out_grad in out_grads
            ]
        return out, out_grads

    def compare_paddle_res_to_torch(self):
        paddle.disable_static(paddle.CUDAPlace(0))
        self.gen_np_inputs()
        p_x, p_dout = self.gen_paddle_inputs()
        t_x, t_dout = self.gen_torch_inputs()
        p_out, p_grads = self.cal_paddle_res(p_x, p_dout)
        t_out, t_grads = self.cal_torch_res(t_x, t_dout)
        np.testing.assert_allclose(
            p_out.numpy(),
            t_out.detach().cpu().numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )
        for idx in range(len(p_grads)):
            np.testing.assert_allclose(
                p_grads[idx].numpy(),
                t_grads[idx].detach().cpu().numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )

        out_static, grads_static = self.compare_static_paddle_res_to_torch()
        np.testing.assert_allclose(
            out_static,
            t_out.detach().cpu().numpy(),
            rtol=self.rtol,
            atol=self.atol,
        )
        for idx in range(len(grads_static)):
            np.testing.assert_allclose(
                grads_static[idx],
                t_grads[idx].detach().cpu().numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )

    def compare_static_paddle_res_to_torch(self):
        paddle.enable_static()
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            x_static, dout_static = self.gen_paddle_static_inputs()
            (out_static, grads_static) = self.cal_paddle_static_res(
                x_static, dout_static
            )
        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(sp)
        out = exe.run(
            mp,
            feed={"x": self.np_x, "dout": self.np_dout},
            fetch_list=[out_static] + grads_static,
        )
        out_static, grads_static = out[0], out[1:]

        return out_static, grads_static

    # 0D Tensor
    def test_case0(self):
        self.shape = []
        self.axis = -1
        self.compare_paddle_res_to_torch()

    # axis != -1
    def test_case1_1(self):
        self.shape = [10, 20, 30, 40]
        self.axis = 1
        self.compare_paddle_res_to_torch()

    def test_case1_2(self):
        self.shape = [10, 20, 30, 1025]
        self.axis = -1
        self.compare_paddle_res_to_torch()

    # 关闭cudnn
    def test_case1_3(self):
        self.shape = [2, 2, 3, 110000]
        self.axis = -1
        self.compare_paddle_res_to_torch()

    def test_case1_4(self):
        self.axis = -1
        for i in range(11):
            self.shape = [2, 3, 2**i]
            self.compare_paddle_res_to_torch()

    def test_case1_5(self):
        self.axis = -1
        for i in range(11):
            self.shape = [2, 3, (2**i + 2)]
            self.compare_paddle_res_to_torch()

    def test_case1_6(self):
        self.axis = -1
        for i in range(11):
            self.shape = [2, 3, (2**i + 1)]
            self.compare_paddle_res_to_torch()

    # #axis != -1
    # def test_case2_1(self):
    #     self.shape = [10, 20, 110000, 100]
    #     self.axis = 1
    #     self.compare_paddle_res_to_torch()

    # def test_case2_2(self):
    #     self.shape = [10, 20, 11000, 1025]
    #     self.axis = -1
    #     self.compare_paddle_res_to_torch()

    # #关闭cudnn
    # def test_case2_3(self):
    #     self.shape = [10, 20, 100, 110000]
    #     self.axis = -1
    #     self.compare_paddle_res_to_torch()

    # def test_case2_4(self):
    #     self.axis = -1
    #     for i in range(11):
    #         self.shape = [2200, 1000, 2**(10-i),2**i]
    #         self.compare_paddle_res_to_torch()

    # def test_case2_5(self):
    #     self.axis = -1
    #     for i in range(11):
    #         self.shape = [2200, 1000, 2**(10-i), (2**i + 2)]
    #         self.compare_paddle_res_to_torch()

    # def test_case2_6(self):
    #     self.axis = -1
    #     for i in range(11):
    #         self.shape = [2200, 1000, 2**(10-i), (2**i + 1)]
    #         self.compare_paddle_res_to_torch()


class TestLogSoftmax_FP16(TestLogSoftmax_FP32):
    def setUp(self):
        self.dtype = "float16"
        self.rtol = 1e-4
        self.atol = 1e-4


class TestLogSoftmax_BF16(TestLogSoftmax_FP32):
    def setUp(self):
        self.dtype = "bfloat16"
        self.rtol = 1e-3
        self.atol = 1e-3


if __name__ == "__main__":
    unittest.main()
