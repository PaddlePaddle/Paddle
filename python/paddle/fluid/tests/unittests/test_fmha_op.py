#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid

paddle.enable_static()


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def ref_softmax(x, axis=None, dtype=None):
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    return np.apply_along_axis(stable_softmax, axis, x_t)


class TestFMHAOp(OpTest):
    def setUp(self):
        self.op_type = "fmha"
        self.dtype = np.float16
        self.init_dtype_type()
        self.place = fluid.CUDAPlace(0)
        # self.place = fluid.core_avx.CPUPlace()
        self.inputs = {
            'X': np.random.random((128, 3, 1024)).astype(self.dtype),
            'Seqlen': np.array(
                [128], dtype=np.int32),
            'Cu_seqlen': np.array(
                [0, 128], dtype=np.int32)
        }

        qkv = self.inputs['X'].reshape([128, 3, 16, 64]).astype(np.float32)
        q = qkv[:, 0, :, :].transpose([1, 0, 2])
        k = qkv[:, 1, :, :].transpose([1, 0, 2])
        v = qkv[:, 2, :, :].transpose([1, 0, 2])

        qk = np.matmul(q, k.transpose([0, 2, 1]))
        scale_qk = qk * (1 / 8)
        softmax_qk = ref_softmax(scale_qk, axis=-1)
        qkv = np.matmul(softmax_qk, v)
        out = qkv.transpose([1, 0, 2]).reshape([128, 1024]).astype(np.float16)

        self.outputs = {'Out': out, 'SoftmaxMask': np.zeros((16, 128, 128))}
        self.grad_x = np.zeros((128, 3, 1024), dtype=np.float16)
        self.grad_out = np.zeros((128, 1024), dtype=np.float16)
        self.grad_softmax_mask = np.zeros((16, 128, 128), dtype=np.float16)

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-1)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['X'], ['Out', 'SoftmaxMask'],
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out, self.grad_softmax_mask],
            check_dygraph=False)
