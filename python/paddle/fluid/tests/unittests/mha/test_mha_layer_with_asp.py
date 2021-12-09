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
from paddle import fluid
from paddle.nn import cuDNNMultiHeadAttention
from paddle.nn.layer import cuDNNSeqInfoInfer
from utils import generate_weight, generate_data


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestcuDNNMultiHeadAttentionStaticWithASP(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.nheads = 4
        self.seqlen = 4
        self.embed_dim = 8

        paddle.enable_static()

        self.place = core.CUDAPlace(0)
        self.exe = paddle.static.Executor(self.place)
        self.dtype = np.single

        self.query, self.key, self.value = generate_data(
            self.batch_size, self.seqlen, self.embed_dim, self.dtype)
        self.weight, _, _, _, _, _, _, _, _, = generate_weight(self.embed_dim,
                                                               self.dtype)

        self.cudnn_main_prog = paddle.static.Program()
        self.cudnn_startup_prog = paddle.static.Program()

        with paddle.static.program_guard(self.cudnn_main_prog,
                                         self.cudnn_startup_prog):
            q_input = paddle.static.data(
                name="q_input",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            k_input = paddle.static.data(
                name="k_input",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            v_input = paddle.static.data(
                name="v_input",
                shape=[-1, self.seqlen, self.embed_dim],
                dtype='float32')
            attn_mask_input = paddle.static.data(
                name="attn_mask", shape=[-1, self.seqlen], dtype="int32")

            q_input.stop_gradient = False
            k_input.stop_gradient = False
            v_input.stop_gradient = False

            seq_info_infer = cuDNNSeqInfoInfer()
            self.cudnn_mha = cuDNNMultiHeadAttention(self.embed_dim,
                                                     self.nheads)
            seq_info = seq_info_infer(attn_mask_input)
            cudnn_mha_output = self.cudnn_mha(q_input, k_input, v_input,
                                              seq_info)
            loss = paddle.mean(cudnn_mha_output)
            optimizer = fluid.contrib.mixed_precision.decorator.decorate(
                fluid.optimizer.SGD(learning_rate=0.01))
            optimizer = fluid.contrib.sparsity.decorate(optimizer)
            optimizer.minimize(loss, self.cudnn_startup_prog)

        self.exe.run(self.cudnn_startup_prog)

        self.cudnn_mha.weight.set_value(self.weight)
        self.attn_mask = np.ones((self.batch_size, self.seqlen), dtype=np.int32)

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
                             "q_input": self.query,
                             "k_input": self.key,
                             "v_input": self.value,
                             "attn_mask": self.attn_mask
                         })

        param_name = "{}.w_0".format(self.cudnn_mha.full_name())
        cudnn_weight = np.array(fluid.global_scope().find_var(param_name)
                                .get_tensor())
        for w in self._get_cudnn_mha_weight(cudnn_weight):
            self.assertTrue(fluid.contrib.sparsity.check_sparsity(w.T))

    def _get_cudnn_mha_weight(self, weight):
        q_proj_weight, _, k_proj_weight, _, \
        v_proj_weight, _, out_proj_weight, _ = \
            cuDNNMultiHeadAttention._split_weight_into_legacy_format(weight, self.embed_dim, False)
        return q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight
