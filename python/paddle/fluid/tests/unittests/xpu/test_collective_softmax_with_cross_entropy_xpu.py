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
import sys
import unittest

import numpy as np
from test_collective_base_xpu import DataTypeCast, TestDistBase

import paddle
from paddle.framework import core

sys.path.append("..")

from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

paddle.enable_static()


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
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


def softmax_with_cross_entropy_grad(softmax, label, loss_grad, axis):
    logit_grad = softmax.copy()
    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    d = int(np.prod(shape[axis:]))
    for i in range(n * d):
        row = int(i / d)
        col = i % d
        if col == label[row]:
            logit_grad[row][col] = (logit_grad[row][col] - 1.0) * loss_grad[row]
        else:
            logit_grad[row][col] = logit_grad[row][col] * loss_grad[row]
    return logit_grad


class XPUTestCSoftmaxWithCEOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'c_softmax_with_cross_entropy'
        self.use_dynamic_create_class = False

    class TestCSoftmaxWithCEOp(TestDistBase):
        def _setup_config(self):
            pass

        def test_softmax_with_ce(self):
            self.batch_size = 10
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
            data_type,
            check_error_log=False,
            need_envs={},
        ):
            required_envs = {
                "FLAGS_eager_delete_tensor_gb": "0.0",
                "PATH": os.getenv("PATH"),
                "PYTHONPATH": os.getenv("PYTHONPATH", ""),
                "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
                "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
                "GLOG_v": "0",
                "DATA_TYPE": data_type,
            }
            required_envs.update(need_envs)
            if check_error_log:
                required_envs["GLOG_v"] = "3"
                required_envs["GLOG_logtostderr"] = "1"
            np_data_type = DataTypeCast(data_type)

            tr0_out, tr1_out, pid0, pid1 = self._run_cluster(
                model_file, required_envs
            )

            # get data that is shared by both ranks
            np.random.seed(os.getuid())
            label = np.random.randint(
                0, self.num_class, size=(self.batch_size, 1), dtype='int32'
            )
            loss_grad = np.random.uniform(
                low=-10.0, high=10.0, size=(self.batch_size, 1)
            ).astype(np_data_type)

            local_elements = int(self.num_class / 2)
            # get input data for rank 0
            np.random.seed(pid0)
            input0 = np.random.uniform(
                low=-10.0, high=10.0, size=(self.batch_size, local_elements)
            ).astype(np_data_type)

            # get input data for rank 1
            np.random.seed(pid1)
            input1 = np.random.uniform(
                low=-10.0, high=10.0, size=(self.batch_size, local_elements)
            ).astype(np_data_type)

            # get combined input data
            inputs = np.concatenate((input0, input1), axis=1)

            # calculate analytic result
            need_softmax = np.apply_along_axis(stable_softmax, 1, inputs)
            need_loss = cross_entropy(need_softmax, label, False, 1)
            need_logits_grad = softmax_with_cross_entropy_grad(
                need_softmax, label, loss_grad, axis=1
            )

            # get real result
            loss0, softmax0, logits_grad0 = tr0_out
            loss1, softmax1, logits_grad1 = tr1_out
            softmax = np.concatenate((softmax0, softmax1), axis=1)
            logits_grad = np.concatenate((logits_grad0, logits_grad1), axis=1)

            # compare results
            rtol = 1e-6
            np.testing.assert_allclose(loss0, need_loss, rtol=rtol)
            np.testing.assert_allclose(loss1, need_loss, rtol=rtol)
            np.testing.assert_allclose(softmax, need_softmax, rtol=rtol)
            np.testing.assert_allclose(logits_grad, need_logits_grad, rtol=rtol)


support_types = get_xpu_op_support_types('c_softmax_with_cross_entropy')
for stype in support_types:
    create_test_class(
        globals(),
        XPUTestCSoftmaxWithCEOP,
        stype,
        ignore_device_version=[core.XPUVersion.XPU1],
    )

if __name__ == '__main__':
    unittest.main()
