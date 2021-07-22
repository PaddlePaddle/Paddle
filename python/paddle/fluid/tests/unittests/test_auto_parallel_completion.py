# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
import paddle.tensor as tensor
import paddle.distributed.auto_parallel as auto
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list

paddle.enable_static()


def compare_program(src_prog, dst_prog):
    """Compare program to check whether they are same."""

    if src_prog.num_blocks != dst_prog.num_blocks:
        print(
            "Block number of src_program {} is not equal to that of dst_program {}."
            .format(src_prog.num_blocks, dst_prog.num_blocks))
        return False

    for src_block, dst_block in zip(src_prog.blocks, dst_prog.blocks):
        # compare vars from src_block and dst_block
        if len(src_block.vars) != len(dst_block.vars):
            print(
                "The number of variables in src_block {} is not equal to that in dst_block {}."
                .format(src_block.idx, dst_block.idx))
            return False
        for src_var_name, src_var_value in src_block.vars.items():
            dst_var_value = dst_block.vars.get(src_var_name)
            if dst_var_value is None:
                print(
                    "The variable {} from src_block doesn't exist in dst_block.".
                    format(src_var_name))
                return False
            if src_var_value.to_string(True, True) != dst_var_value.to_string(
                    True, True):
                print(
                    "The variable {} of src_block is not equal to variable {} of dst_block."
                    .format(src_var_name, src_var_name))
                return False

        # compare ops from src_block and dst_block
        if len(src_block.ops) != len(dst_block.ops):
            print(
                "The number of operators in src_block {} is not equal to that in dst_block {}."
                .format(src_block.idx, dst_block.idx))
        for src_op, dst_op in zip(src_block.ops, dst_block.ops):
            if src_op.type != dst_op.type:
                print(
                    "The operator's type {} of src_block is not equal to the operator'type {} of dst_block."
                    .format(src_op.type, dst_op.type))
            src_op_callstack = src_op.attr("op_callstack")
            dst_op_callstack = dst_op.attr("op_callstack")
            # print(src_op_callstack, dst_op_callstack)
            src_op._remove_attr("op_callstack")
            dst_op._remove_attr("op_callstack")
            if src_op.to_string(True) != dst_op.to_string(True):
                print(
                    "The operator {}'s content of src_block is not equal to the operator {}'s content of dst_block."
                    .format(src_op.type, dst_op.type))
                # print(src_op.to_string(True), dst_op.to_string(True))
                return False
            else:
                src_op._set_attr("op_callstack", src_op_callstack)
                dst_op._set_attr("op_callstack", dst_op_callstack)

        return True


class TestMLPAutoCompletion(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.hidden_size = 1024
        self.sequence_len = 128
        self.dropout_ratio = 0.1
        self.initializer_range = 0.02

        self.train_prog = static.Program()
        self.start_prog = static.Program()
        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            intermediate_size = 4 * self.hidden_size
            d_model = self.hidden_size
            dim_feedforward = intermediate_size
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=self.initializer_range))
            bias_attr = None

            self.linear3 = nn.Linear(
                d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
            self.linear4 = nn.Linear(
                dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
            self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
            self.dropout = nn.Dropout(
                self.dropout_ratio, mode="upscale_in_train")

    def test_mlp_dp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim(
        ) == 1, "The number dimension of process mesh must to be 1"
        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            input = static.data(
                name="input",
                shape=[self.batch_size, self.sequence_len, self.hidden_size],
                dtype='float32')
            out0 = self.norm(input)
            out1 = self.linear3(out0)
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear4(out2)
            out4 = self.dropout(out3)
            auto.shard_tensor(out4, proc_mesh, dims_mapping=[0, -1, -1])
        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)

    def test_mlp_mp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim(
        ) == 1, "The number dimension of process mesh must to be 1"
        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            input = static.data(
                name="input",
                shape=[self.batch_size, self.sequence_len, self.hidden_size],
                dtype='float32')
            out0 = self.norm(input)
            out1 = self.linear3(out0)
            auto.shard_tensor(
                self.linear3.weight, proc_mesh, dims_mapping=[-1, 0])
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear4(out2)
            auto.shard_tensor(
                self.linear4.weight, proc_mesh, dims_mapping=[0, -1])
            out4 = self.dropout(out3)
        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)

    def test_mlp_dp_mp(self):
        proc_mesh = auto.ProcessMesh(
            shape=[2, 4], process_group=[0, 1, 2, 3, 4, 5, 6, 7])
        assert proc_mesh.get_ndim(
        ) == 2, "The number dimension of process mesh must to be 2"
        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            input = static.data(
                name="input",
                shape=[self.batch_size, self.sequence_len, self.hidden_size],
                dtype='float32')
            out0 = self.norm(input)
            auto.shard_tensor(out0, proc_mesh, dims_mapping=[0, -1, -1])
            out1 = self.linear3(out0)
            auto.shard_tensor(
                self.linear3.weight, proc_mesh, dims_mapping=[-1, 1])
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear4(out2)
            auto.shard_tensor(
                self.linear4.weight, proc_mesh, dims_mapping=[1, -1])
            out4 = self.dropout(out3)
        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)


class TestAttentionAutoCompletion(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.hidden_size = 1024
        self.sequence_len = 128
        self.embed_dim = self.hidden_size
        self.kdim = self.embed_dim
        self.vdim = self.embed_dim
        self.num_heads = 8
        self.dropout_ratio = 0.1
        self.initializer_range = 0.02
        self.training = True
        self.attn_mask = None

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.train_prog = static.Program()
        self.start_prog = static.Program()
        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            self.input = static.data(
                name="query",
                shape=[self.batch_size, self.sequence_len, self.hidden_size],
                dtype='float32')
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=self.initializer_range))
            bias_attr = None
            self.q_proj = nn.Linear(
                self.embed_dim,
                self.embed_dim,
                weight_attr,
                bias_attr=bias_attr)
            self.k_proj = nn.Linear(
                self.kdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
            self.v_proj = nn.Linear(
                self.vdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
            self.out_proj = nn.Linear(
                self.embed_dim,
                self.embed_dim,
                weight_attr,
                bias_attr=bias_attr)

    def test_attn_dp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim(
        ) == 1, "The number dimension of process mesh must to be 1"

        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            auto.shard_tensor(self.input, proc_mesh, dims_mapping=[0, -1, -1])
            q = self.q_proj(self.input)
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

            k = self.k_proj(self.input)
            v = self.v_proj(self.input)
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            # scale dot product attention
            product = layers.matmul(
                x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

            if self.attn_mask is not None:
                product = product + self.attn_mask

            weights = F.softmax(product)

            if self.dropout_ratio:
                weights = F.dropout(
                    weights,
                    self.dropout_ratio,
                    training=self.training,
                    mode="upscale_in_train")

            out = tensor.matmul(weights, v)

            # combine heads
            out = tensor.transpose(out, perm=[0, 2, 1, 3])
            out = tensor.reshape(
                x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

            # project to output
            out = self.out_proj(out)

        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)

    def test_attn_mp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim(
        ) == 1, "The number dimension of process mesh must to be 1"

        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            q = self.q_proj(self.input)
            auto.shard_tensor(
                self.q_proj.weight, proc_mesh, dims_mapping=[-1, 0])
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

            k = self.k_proj(self.input)
            auto.shard_tensor(
                self.k_proj.weight, proc_mesh, dims_mapping=[-1, 0])
            v = self.v_proj(self.input)
            auto.shard_tensor(
                self.v_proj.weight, proc_mesh, dims_mapping=[-1, 0])
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            # scale dot product attention
            product = layers.matmul(
                x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

            if self.attn_mask is not None:
                product = product + self.attn_mask

            weights = F.softmax(product)

            if self.dropout_ratio:
                weights = F.dropout(
                    weights,
                    self.dropout_ratio,
                    training=self.training,
                    mode="upscale_in_train")

            out = tensor.matmul(weights, v)

            # combine heads
            out = tensor.transpose(out, perm=[0, 2, 1, 3])
            out = tensor.reshape(
                x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

            # project to output
            out = self.out_proj(out)
            auto.shard_tensor(
                self.out_proj.weight, proc_mesh, dims_mapping=[0, -1])

        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)

    def test_attn_dp_mp(self):
        proc_mesh = auto.ProcessMesh(
            shape=[2, 4], process_group=[0, 1, 2, 3, 4, 5, 6, 7])
        assert proc_mesh.get_ndim(
        ) == 2, "The number dimension of process mesh must to be 2"

        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            auto.shard_tensor(self.input, proc_mesh, dims_mapping=[0, -1, -1])
            q = self.q_proj(self.input)
            auto.shard_tensor(
                self.q_proj.weight, proc_mesh, dims_mapping=[-1, 1])
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

            k = self.k_proj(self.input)
            auto.shard_tensor(
                self.k_proj.weight, proc_mesh, dims_mapping=[-1, 1])
            v = self.v_proj(self.input)
            auto.shard_tensor(
                self.v_proj.weight, proc_mesh, dims_mapping=[-1, 1])
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            # scale dot product attention
            product = layers.matmul(
                x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

            if self.attn_mask is not None:
                product = product + self.attn_mask

            weights = F.softmax(product)

            if self.dropout_ratio:
                weights = F.dropout(
                    weights,
                    self.dropout_ratio,
                    training=self.training,
                    mode="upscale_in_train")

            out = tensor.matmul(weights, v)

            # combine heads
            out = tensor.transpose(out, perm=[0, 2, 1, 3])
            out = tensor.reshape(
                x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

            # project to output
            out = self.out_proj(out)
            auto.shard_tensor(
                self.out_proj.weight, proc_mesh, dims_mapping=[1, -1])

        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)


class TestTransformerDecoderLayerAutoCompletion(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.hidden_size = 1024
        self.sequence_len = 128
        self.embed_dim = self.hidden_size
        self.kdim = self.embed_dim
        self.vdim = self.embed_dim
        self.num_heads = 8
        self.dropout_ratio = 0.1
        self.initializer_range = 0.02
        self.training = True
        self.attn_mask = None

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"

        self.train_prog = static.Program()
        self.start_prog = static.Program()
        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            self.input = static.data(
                name="query",
                shape=[self.batch_size, self.sequence_len, self.hidden_size],
                dtype='float32')
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=self.initializer_range))
            bias_attr = None
            self.q_proj = nn.Linear(
                self.embed_dim,
                self.embed_dim,
                weight_attr,
                bias_attr=bias_attr)
            self.k_proj = nn.Linear(
                self.kdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
            self.v_proj = nn.Linear(
                self.vdim, self.embed_dim, weight_attr, bias_attr=bias_attr)
            self.out_proj = nn.Linear(
                self.embed_dim,
                self.embed_dim,
                weight_attr,
                bias_attr=bias_attr)

            intermediate_size = 4 * self.hidden_size
            d_model = self.hidden_size
            dim_feedforward = intermediate_size
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=self.initializer_range))
            bias_attr = None
            weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
            bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

            self.linear0 = nn.Linear(
                d_model,
                dim_feedforward,
                weight_attrs[2],
                bias_attr=bias_attrs[2])
            self.linear1 = nn.Linear(
                dim_feedforward,
                d_model,
                weight_attrs[2],
                bias_attr=bias_attrs[2])
            self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
            self.dropout = nn.Dropout(
                self.dropout_ratio, mode="upscale_in_train")

    def test_decoder_dp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim(
        ) == 1, "The number dimension of process mesh must to be 1"

        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            auto.shard_tensor(self.input, proc_mesh, dims_mapping=[0, -1, -1])
            # Pre-norm
            target = self.norm(self.input)

            # The following is the attention part
            q = self.q_proj(target)
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

            k = self.k_proj(target)
            v = self.v_proj(target)
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            # scale dot product attention
            product = layers.matmul(
                x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

            if self.attn_mask is not None:
                product = product + self.attn_mask

            weights = F.softmax(product)

            if self.dropout_ratio:
                weights = F.dropout(
                    weights,
                    self.dropout_ratio,
                    training=self.training,
                    mode="upscale_in_train")

            out = tensor.matmul(weights, v)

            # combine heads
            out = tensor.transpose(out, perm=[0, 2, 1, 3])
            out = tensor.reshape(
                x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

            # project to output
            out = self.out_proj(out)

            # Add residual
            residual = self.input + self.dropout(out)

            # Pre-norm
            out0 = self.norm(residual)

            # The following is the MLP part
            out1 = self.linear0(out0)
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear1(out2)

            # Add residual
            final = residual + self.dropout(out3)

        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)

    def test_decoder_mp(self):
        proc_mesh = auto.ProcessMesh(shape=[4], process_group=[0, 1, 2, 3])
        assert proc_mesh.get_ndim(
        ) == 1, "The number dimension of process mesh must to be 1"

        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            # Pre-norm
            target = self.norm(self.input)

            # The following is the attention part
            q = self.q_proj(target)
            auto.shard_tensor(
                self.q_proj.weight, proc_mesh, dims_mapping=[-1, 0])
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

            k = self.k_proj(target)
            auto.shard_tensor(
                self.k_proj.weight, proc_mesh, dims_mapping=[-1, 0])
            v = self.v_proj(target)
            auto.shard_tensor(
                self.v_proj.weight, proc_mesh, dims_mapping=[-1, 0])
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            # scale dot product attention
            product = layers.matmul(
                x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

            if self.attn_mask is not None:
                product = product + self.attn_mask

            weights = F.softmax(product)

            if self.dropout_ratio:
                weights = F.dropout(
                    weights,
                    self.dropout_ratio,
                    training=self.training,
                    mode="upscale_in_train")

            out = tensor.matmul(weights, v)

            # combine heads
            out = tensor.transpose(out, perm=[0, 2, 1, 3])
            out = tensor.reshape(
                x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

            # project to output
            out = self.out_proj(out)
            auto.shard_tensor(
                self.out_proj.weight, proc_mesh, dims_mapping=[0, -1])

            # Add residual
            residual = self.input + self.dropout(out)

            # Pre-norm
            out0 = self.norm(residual)

            # The following is the MLP part
            out1 = self.linear0(out0)
            auto.shard_tensor(
                self.linear0.weight, proc_mesh, dims_mapping=[-1, 0])
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear1(out2)
            auto.shard_tensor(
                self.linear1.weight, proc_mesh, dims_mapping=[0, -1])

            # Add residual
            final = residual + self.dropout(out3)

        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)

    def test_decoder_dp_mp(self):
        proc_mesh = auto.ProcessMesh(
            shape=[2, 4], process_group=[0, 1, 2, 3, 4, 5, 6, 7])
        assert proc_mesh.get_ndim(
        ) == 2, "The number dimension of process mesh must to be 2"

        with static.program_guard(self.train_prog, self.start_prog), utils.unique_name.guard():
            auto.shard_tensor(self.input, proc_mesh, dims_mapping=[0, -1, -1])
            # Pre-norm
            target = self.norm(self.input)

            # The following is the attention part
            q = self.q_proj(target)
            auto.shard_tensor(
                self.q_proj.weight, proc_mesh, dims_mapping=[-1, 1])
            q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
            q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

            k = self.k_proj(target)
            auto.shard_tensor(
                self.k_proj.weight, proc_mesh, dims_mapping=[-1, 1])
            v = self.v_proj(target)
            auto.shard_tensor(
                self.v_proj.weight, proc_mesh, dims_mapping=[-1, 1])
            k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
            k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
            v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
            v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

            # scale dot product attention
            product = layers.matmul(
                x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

            if self.attn_mask is not None:
                product = product + self.attn_mask

            weights = F.softmax(product)

            if self.dropout_ratio:
                weights = F.dropout(
                    weights,
                    self.dropout_ratio,
                    training=self.training,
                    mode="upscale_in_train")

            out = tensor.matmul(weights, v)

            # combine heads
            out = tensor.transpose(out, perm=[0, 2, 1, 3])
            out = tensor.reshape(
                x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

            # project to output
            out = self.out_proj(out)
            auto.shard_tensor(
                self.out_proj.weight, proc_mesh, dims_mapping=[1, -1])

            # Add residual
            residual = self.input + self.dropout(out)

            # Pre-norm
            out0 = self.norm(residual)

            # The following is the MLP part
            out1 = self.linear0(out0)
            auto.shard_tensor(
                self.linear0.weight, proc_mesh, dims_mapping=[-1, 1])
            out2 = F.gelu(out1, approximate=True)
            out3 = self.linear1(out2)
            auto.shard_tensor(
                self.linear1.weight, proc_mesh, dims_mapping=[1, -1])

            # Add residual
            final = residual + self.dropout(out3)

        print(self.train_prog)
        complete_prog = auto.complete_annotation(self.train_prog)
        print(complete_prog)


if __name__ == "__main__":
    unittest.main()
