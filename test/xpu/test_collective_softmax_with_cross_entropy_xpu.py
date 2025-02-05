#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_uint16_to_float
from test_collective_base_xpu import DataTypeCast, TestDistBase

import paddle
from paddle.framework import core

paddle.enable_static()


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1 :]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


def softmax_with_cross_entropy_grad(
    softmax, label, loss_grad, axis, ignore_index=-1
):
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    d = int(np.prod(shape[axis:]))
    logit_grad_2d = softmax.copy().reshape(n, d)
    loss_grad_2d = loss_grad.reshape(n, 1)
    label_2d = label.reshape(n, 1)
    for i in range(n * d):
        row = int(i / d)
        col = i % d
        if label_2d[row] == ignore_index:
            logit_grad_2d[row][col] = 0
        else:
            if col == label_2d[row]:
                logit_grad_2d[row][col] = (
                    logit_grad_2d[row][col] - 1.0
                ) * loss_grad_2d[row]
            else:
                logit_grad_2d[row][col] = (
                    logit_grad_2d[row][col] * loss_grad_2d[row]
                )
    logit_grad = logit_grad_2d.reshape(softmax.shape)
    return logit_grad


class XPUTestCSoftmaxWithCEOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'c_softmax_with_cross_entropy'
        self.use_dynamic_create_class = False

    class TestCSoftmaxWithCEOp(TestDistBase):
        def _setup_config(self):
            pass

        def test_softmax_with_ce_2d_logits(self):
            self.batch_size = 1
            self.seq_len = 10
            self.num_class = 1000
            self.check_with_place(
                "collective_softmax_with_cross_entropy_op_xpu.py",
                "softmax_with_ce",
                self.in_type_str,
            )

        def check_with_place(
            self,
            model_file,
            col_type,
            dtype=None,
            check_error_log=False,
            need_envs={},
        ):
            required_envs = {
                "FLAGS_eager_delete_tensor_gb": "0.0",
                "PATH": os.getenv("PATH"),
                "PYTHONPATH": os.getenv("PYTHONPATH", ""),
                "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
                "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
                "GLOG_v": "3",
                "DTYPE": dtype,
                "BATCH_SIZE": str(self.batch_size),
            }
            required_envs.update(need_envs)
            if check_error_log:
                required_envs["GLOG_v"] = "3"
                required_envs["GLOG_logtostderr"] = "1"
            np_dtype = DataTypeCast(dtype)

            tr0_out, tr1_out, pid0, pid1 = self._run_cluster(
                model_file, required_envs
            )

            # if batch_size = 1, we treat logits/labels as 2D tensors
            # if batch_size > 1, we treat logits/labels as 3D tensors
            local_elements = int(self.num_class / 2)
            if self.batch_size > 1:
                logits_shape = [self.batch_size, self.seq_len, local_elements]
                label_shape = [self.batch_size, self.seq_len, 1]
            else:
                logits_shape = [self.seq_len, local_elements]
                label_shape = [self.seq_len, 1]

            # get data that is shared by both ranks
            np.random.seed(os.getuid())
            label = np.random.randint(
                0, self.num_class, size=label_shape, dtype='int32'
            )
            ignore_index = label[0][0]
            loss_grad = np.random.uniform(
                low=-10.0, high=10.0, size=label_shape
            ).astype(np_dtype)

            # get input data for rank 0
            np.random.seed(pid0)
            input0 = np.random.uniform(
                low=-40.0, high=40.0, size=logits_shape
            ).astype(np_dtype)

            # get input data for rank 1
            np.random.seed(pid1)
            input1 = np.random.uniform(
                low=-40.0, high=40.0, size=logits_shape
            ).astype(np_dtype)

            # get combined input data
            inputs = np.concatenate((input0, input1), axis=-1)

            # calculate analytic result
            need_softmax = np.apply_along_axis(stable_softmax, -1, inputs)
            need_loss = cross_entropy(
                need_softmax, label, False, -1, ignore_index
            )
            need_logits_grad = softmax_with_cross_entropy_grad(
                need_softmax,
                label,
                loss_grad,
                axis=-1,
                ignore_index=ignore_index,
            )

            # get real result
            loss0, softmax0, logits_grad0 = tr0_out
            loss1, softmax1, logits_grad1 = tr1_out
            if dtype == "bfloat16":
                loss0 = convert_uint16_to_float(loss0)
                softmax0 = convert_uint16_to_float(softmax0)
                logits_grad0 = convert_uint16_to_float(logits_grad0)
                loss1 = convert_uint16_to_float(loss1)
                softmax1 = convert_uint16_to_float(softmax1)
                logits_grad1 = convert_uint16_to_float(logits_grad1)
            softmax = np.concatenate((softmax0, softmax1), axis=-1)
            logits_grad = np.concatenate((logits_grad0, logits_grad1), axis=-1)

            # compare results
            rtol = 1e-6
            atol = 0
            if dtype == "bfloat16":
                rtol = 0.1
                atol = 0.1
            np.testing.assert_allclose(loss0, need_loss, rtol=rtol, atol=atol)
            np.testing.assert_allclose(loss1, need_loss, rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                softmax, need_softmax, rtol=rtol, atol=atol
            )
            np.testing.assert_allclose(
                logits_grad, need_logits_grad, rtol=rtol, atol=atol
            )

    class TestCSoftmaxWithCEOp1(TestCSoftmaxWithCEOp):
        def _setup_config(self):
            pass

        def test_softmax_with_ce_3d_logis(self):
            self.batch_size = 2
            self.seq_len = 10
            self.num_class = 1000
            self.check_with_place(
                "collective_softmax_with_cross_entropy_op_xpu.py",
                "softmax_with_ce",
                self.in_type_str,
            )


support_types = get_xpu_op_support_types('c_softmax_with_cross_entropy')
for stype in support_types:
    create_test_class(
        globals(),
        XPUTestCSoftmaxWithCEOP,
        stype,
        ignore_device_version=[core.XPUVersion.XPU1, core.XPUVersion.XPU3],
    )

if __name__ == '__main__':
    unittest.main()
