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
    elif dtype in ['int64', np.int64]:
        return torch.int64
    elif dtype in ['int32', np.int32]:
        return torch.int32
    elif dtype in ['int16', np.int16]:
        return torch.int16
    elif dtype in ['int8', np.int8]:
        return torch.int8
    elif dtype in ['uint8', np.uint8]:
        return torch.uint8


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def ref_softmax(x, axis=None, dtype=None):
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    return np.apply_along_axis(stable_softmax, axis, x_t)


class TestCrossEntropy_FP32(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.rtol = 1e-6
        self.atol = 1e-6

    def gen_np_inputs_and_dout(self):
        self.logits = np.random.random(size=self.shape).astype("float32") - 0.5
        if not self.use_softmax:
            self.logits_softmax = ref_softmax(self.logits, axis=self.axis)
        if self.soft_label:
            self.labels = np.random.random(self.shape).astype("float32")
            self.labels /= self.labels.sum(axis=-1, keepdims=True)
        else:
            self.labels = np.random.randint(
                low=0, high=self.shape[-1], size=self.shape[:-1]
            ).astype("int64")
        self.dout = np.random.random(size=[]).astype("float32") - 0.5

    def gen_paddle_inputs(self):
        p_logits = paddle.to_tensor(
            self.logits if self.use_softmax else self.logits_softmax,
            dtype=self.dtype,
            place='gpu',
        )
        p_logits.stop_gradient = False
        p_labels = paddle.to_tensor(
            self.labels,
            dtype=self.dtype if self.soft_label else self.hard_label_dtype,
            place='gpu',
        )
        p_dout = paddle.to_tensor(self.dout, dtype=self.dtype, place='gpu')
        return p_logits, p_labels, p_dout

    def gen_paddle_static_inputs(self):
        p_logits_static = paddle.static.data(
            name="logits", shape=self.shape, dtype=self.dtype
        )
        p_logits_static.stop_gradient = False
        p_labels_static = paddle.static.data(
            name="labels",
            shape=self.shape if self.soft_label else self.shape[:-1],
            dtype=self.dtype if self.soft_label else self.hard_label_dtype,
        )
        p_dout_static = paddle.static.data(
            name="dout", shape=[], dtype=self.dtype
        )
        p_dout_static.stop_gradient = False
        return p_logits_static, p_labels_static, p_dout_static

    def gen_torch_inputs(self):
        t_logits = torch.tensor(
            self.logits,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype),
            requires_grad=True,
        )
        t_labels = torch.tensor(
            self.labels,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.soft_label
            else torch.int64,  # convert_dtype_to_torch_type(self.hard_label_dtype),
            requires_grad=False,
        )
        t_dout = torch.tensor(
            self.dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype),
            requires_grad=True,
        )
        return t_logits, t_labels, t_dout

    def cal_paddle_res(self, input, label, dout):
        out = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean',
            soft_label=self.soft_label,
            axis=self.axis,
            use_softmax=self.use_softmax,
            label_smoothing=self.label_smoothing,
        )
        out_grads = paddle.grad([out], [input], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, "float32")
            out_grads = [paddle.cast(g, "float32") for g in out_grads]
        return out, out_grads

    def cal_paddle_static_res(self, input, label, dout):
        if self.dtype == "bfloat16" or self.dtype == "float16":
            input = paddle.cast(input, dtype=self.dtype)
            label = paddle.cast(label, dtype=self.dtype)
            dout = paddle.cast(dout, dtype=self.dtype)
        out = paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean',
            soft_label=self.soft_label,
            axis=self.axis,
            use_softmax=self.use_softmax,
            label_smoothing=self.label_smoothing,
        )
        out_grads = paddle.static.gradients(out, input, target_gradients=dout)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, "float32")
            out_grads = [paddle.cast(g, "float32") for g in out_grads]
        return out, out_grads

    def cal_torch_res(self, input, label, dout):
        out = torch.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            size_average=None,
            ignore_index=self.ignore_index,
            reduce=None,
            reduction='mean',
            label_smoothing=self.label_smoothing,
        )
        out_grads = torch.autograd.grad([out], [input], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = [g.to(dtype=torch.float32) for g in out_grads]
        return out, out_grads

    def compare_res(self):
        paddle.disable_static(paddle.CUDAPlace(0))
        self.gen_np_inputs_and_dout()
        p_logits, p_labels, p_dout = self.gen_paddle_inputs()
        t_logits, t_labels, t_dout = self.gen_torch_inputs()
        p_out, p_grads = self.cal_paddle_res(p_logits, p_labels, p_dout)
        t_out, t_grads = self.cal_torch_res(t_logits, t_labels, t_dout)

        # np.testing.assert_allclose(p_out.numpy(), t_out.detach().cpu().numpy(), rtol=self.rtol, atol=self.atol)
        # if self.use_softmax:
        #     for idx in range(len(p_grads)):
        #         np.testing.assert_allclose(p_grads[idx].numpy(), t_grads[idx].detach().cpu().numpy(), rtol=self.rtol, atol=self.atol)

        # out_static, grads_static = self.compare_static()
        # np.testing.assert_allclose(out_static, t_out.detach().cpu().numpy(), rtol=self.rtol, atol=self.atol)
        # if self.use_softmax:
        #     for idx in range(len(grads_static)):
        #         np.testing.assert_allclose(grads_static[idx], t_grads[idx].detach().cpu().numpy(), rtol=self.rtol, atol=self.atol)

    def compare_static(self):
        paddle.enable_static()
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            (
                logits_static,
                labels_static,
                dout_static,
            ) = self.gen_paddle_static_inputs()
            (out_static, grads_static) = self.cal_paddle_static_res(
                logits_static, labels_static, dout_static
            )
        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(sp)
        out = exe.run(
            mp,
            feed={
                "logits": self.logits
                if self.use_softmax
                else self.logits_softmax,
                "labels": self.labels,
                "dout": self.dout,
            },
            fetch_list=[out_static] + grads_static,
        )
        out_static, grads_static = out[0], out[1:]

        return out_static, grads_static

    def test_case0(self):
        self.shape = [100, 1]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = True
        self.use_softmax = False
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case1(self):
        self.shape = [40, 60]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = True
        self.use_softmax = False
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case2(self):
        self.shape = [100, 1]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = True
        self.use_softmax = True
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case3(self):
        self.shape = [40, 60]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = True
        self.use_softmax = True
        self.label_smoothing = 0.0
        self.compare_res()

    # torch不支持axis参数
    def test_case4(self):
        self.shape = [20, 400]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = True
        self.use_softmax = True
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case5(self):
        self.shape = [100, 1]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = False
        self.hard_label_dtype = "int64"
        self.use_softmax = False
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case6(self):
        self.shape = [100, 200]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = False
        self.hard_label_dtype = "int64"
        self.use_softmax = False
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case7(self):
        self.shape = [100, 1]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = False
        self.hard_label_dtype = "int64"
        self.use_softmax = True
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case8(self):
        self.shape = [100, 200]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = False
        self.hard_label_dtype = "int64"
        self.use_softmax = True
        self.label_smoothing = 0.0
        self.compare_res()

    def test_case9(self):
        self.shape = [100, 400]
        self.axis = -1
        self.weight = None
        self.ignore_index = -100
        self.soft_label = False
        self.hard_label_dtype = "int64"
        self.use_softmax = True
        self.label_smoothing = 0.0
        self.compare_res()

    # def test_case10(self):
    #     self.shape = [100, 400]
    #     self.axis = 0
    #     self.weight = None
    #     self.ignore_index = -100
    #     self.soft_label = False
    #     self.use_softmax = True
    #     self.label_smoothing = 0.0
    #     self.compare_res()


# class TestCrossEntropy_FP16(TestCrossEntropy_FP32):
#     def setUp(self):
#         self.dtype="float16"
#         self.rtol=1e-4
#         self.atol=1e-4

# class TestCrossEntropy_BF16(TestCrossEntropy_FP32):
#     def setUp(self):
#         self.dtype="bfloat16"
#         self.rtol=1e-3
#         self.atol=1e-3

if __name__ == '__main__':
    unittest.main()
