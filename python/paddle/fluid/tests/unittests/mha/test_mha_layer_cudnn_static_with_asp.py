# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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
import paddle.fluid.core as core

import paddle
from paddle.nn import MultiHeadAttention, CUDNNMultiHeadAttention
from paddle.nn.layer import CUDNNSeqInfoInfer
from paddle import fluid


def _generate_data(batch_size, max_seq_len, vec_size):
    Q = (np.random.random(
        (batch_size, max_seq_len, vec_size)) - .5).astype(np.single)
    K = (np.random.random(
        (batch_size, max_seq_len, vec_size)) - .5).astype(np.single)
    V = (np.random.random(
        (batch_size, max_seq_len, vec_size)) - .5).astype(np.single)
    W = (np.random.random((4 * vec_size * vec_size, )) - .5).astype(np.single)
    W = np.concatenate((W, np.zeros((4 * vec_size, ))), dtype=np.single)

    return (Q, K, V, W)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNMHALayerWithASP(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.nheads = 4
        self.seq_len = 4
        self.vec_size = 8

        paddle.enable_static()

        self.place = core.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)

        self.Q, self.K, self.V, self.W = _generate_data(
            self.batch_size, self.seq_len, self.vec_size)

        self.cudnn_main_prog = paddle.static.Program()
        self.cudnn_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.cudnn_main_prog,
                                         self.cudnn_startup_prog):
            q_input = paddle.static.data(
                name="q_input",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            k_input = paddle.static.data(
                name="k_input",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            v_input = paddle.static.data(
                name="v_input",
                shape=[-1, self.seq_len, self.vec_size],
                dtype='float32')
            attn_mask_input = paddle.static.data(
                name="attn_mask", shape=[-1, self.seq_len], dtype="int32")

            q_input.stop_gradient = False
            k_input.stop_gradient = False
            v_input.stop_gradient = False

            seq_info_infer = CUDNNSeqInfoInfer()
            self.cudnn_mha = CUDNNMultiHeadAttention(self.vec_size, self.nheads)
            seq_info = seq_info_infer(attn_mask_input)
            cudnn_mha_output = self.cudnn_mha(q_input, k_input, v_input,
                                              seq_info)
            loss = paddle.mean(cudnn_mha_output)
            optimizer = fluid.contrib.mixed_precision.decorator.decorate(
                fluid.optimizer.SGD(learning_rate=0.01))
            optimizer = fluid.contrib.sparsity.decorate(optimizer)
            optimizer.minimize(loss, self.cudnn_startup_prog)

        self.exe.run(self.cudnn_startup_prog)

        self.cudnn_mha.weight.set_value(self.W)
        self.attn_mask_for_cudnn = np.ones(
            (self.batch_size, self.seq_len), dtype=np.int32)

    def test_cudnn_1D_mask(self):
        param_name = "{}.w_0".format(self.cudnn_mha.full_name())
        pre_pruned_mat = np.array(fluid.global_scope().find_var(param_name)
                                  .get_tensor())
        for w in self._get_cudnn_mha_weight(pre_pruned_mat):
            self.assertFalse(fluid.contrib.sparsity.check_sparsity(w.T))

        fluid.contrib.sparsity.prune_model(
            self.cudnn_main_prog, mask_algo="mask_1d", with_mask=True)

        post_pruned_mat = np.array(fluid.global_scope().find_var(param_name)
                                   .get_tensor())
        for w in self._get_cudnn_mha_weight(post_pruned_mat):
            self.assertTrue(fluid.contrib.sparsity.check_sparsity(w.T))

    def test_cudnn_2D_greedy_mask(self):
        param_name = "{}.w_0".format(self.cudnn_mha.full_name())
        pre_pruned_mat = np.array(fluid.global_scope().find_var(param_name)
                                  .get_tensor())
        for w in self._get_cudnn_mha_weight(pre_pruned_mat):
            self.assertFalse(
                fluid.contrib.sparsity.check_sparsity(
                    w.T, func_name=fluid.contrib.sparsity.CheckMethod.CHECK_2D))

        fluid.contrib.sparsity.prune_model(
            self.cudnn_main_prog, mask_algo="mask_2d_greedy", with_mask=True)

        post_pruned_mat = np.array(fluid.global_scope().find_var(param_name)
                                   .get_tensor())
        for w in self._get_cudnn_mha_weight(post_pruned_mat):
            self.assertTrue(
                fluid.contrib.sparsity.check_sparsity(
                    w.T, func_name=fluid.contrib.sparsity.CheckMethod.CHECK_2D))

    def test_cudnn_2D_best_mask(self):
        self.cudnn_mha.weight.set_value(self.W)
        param_name = "{}.w_0".format(self.cudnn_mha.full_name())
        pre_pruned_mat = np.array(fluid.global_scope().find_var(param_name)
                                  .get_tensor())
        for w in self._get_cudnn_mha_weight(pre_pruned_mat):
            self.assertFalse(
                fluid.contrib.sparsity.check_sparsity(
                    w.T, func_name=fluid.contrib.sparsity.CheckMethod.CHECK_2D))

        fluid.contrib.sparsity.prune_model(
            self.cudnn_main_prog, mask_algo="mask_2d_best", with_mask=True)

        post_pruned_mat = np.array(fluid.global_scope().find_var(param_name)
                                   .get_tensor())
        for w in self._get_cudnn_mha_weight(post_pruned_mat):
            self.assertTrue(
                fluid.contrib.sparsity.check_sparsity(
                    w.T, func_name=fluid.contrib.sparsity.CheckMethod.CHECK_2D))

    def test_asp_workflow(self):
        fluid.contrib.sparsity.prune_model(self.cudnn_main_prog, with_mask=True)
        for _ in range(3):
            self.exe.run(self.cudnn_main_prog,
                         feed={
                             "q_input": self.Q,
                             "k_input": self.K,
                             "v_input": self.V,
                             "attn_mask": self.attn_mask_for_cudnn
                         })

        param_name = "{}.w_0".format(self.cudnn_mha.full_name())
        cudnn_weight = np.array(fluid.global_scope().find_var(param_name)
                                .get_tensor())
        for w in self._get_cudnn_mha_weight(cudnn_weight):
            self.assertTrue(fluid.contrib.sparsity.check_sparsity(w.T))

    def _get_cudnn_mha_weight(self, cudnn_weight):
        param_shape = (self.vec_size, self.vec_size)
        stride = self.vec_size * self.vec_size
        WQ = cudnn_weight[:stride].reshape(param_shape)
        WK = cudnn_weight[stride:2 * stride].reshape(param_shape)
        WV = cudnn_weight[2 * stride:3 * stride].reshape(param_shape)
        WO = cudnn_weight[3 * stride:4 * stride].reshape(param_shape)
        return WQ, WK, WV, WO
