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
import unittest.mock
from io import StringIO
import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddle.distributed.fleet import auto
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.utils import check_distributed_attr_for_program
from paddle.distributed.auto_parallel.utils import print_program_with_dist_attr
from paddle.distributed.auto_parallel.utils import append_distributed_attr_suffix
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.utils import _get_comm_group
from paddle.distributed.auto_parallel.process_group import new_process_group

paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None


def get_programs(annotated_func):
    train_program = static.Program()
    start_program = static.Program()
    dist_context = DistributedContext()
    global _global_process_mesh
    dist_context.process_mesh = _global_process_mesh
    train_program, start_program = annotated_func(train_program, start_program)
    completer = Completer(dist_context)
    complete_train_program = completer.complete_forward_annotation(
        train_program)
    dist_context.block_state.parse_forward_blocks(complete_train_program)

    rank_id = 3
    dist_strategy = fleet.DistributedStrategy()
    partitioner = Partitioner(dist_context, rank_id)
    test_auto_parallel_dist_main_prog, test_auto_parallel_dist_startup_prog, _ = partitioner.partition(
        complete_train_program, start_program, [])

    return complete_train_program, start_program, test_auto_parallel_dist_main_prog, test_auto_parallel_dist_startup_prog, dist_context


def is_all_parameters_shape_equal(prog1, prog2):

    params1 = prog1.all_parameters()
    params2 = prog2.all_parameters()
    params1.sort(key=lambda x: x.name)
    params2.sort(key=lambda x: x.name)
    shape1 = [tensor.shape for tensor in params1]
    shape2 = [tensor.shape for tensor in params2]

    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True


def check_tensor_split(prog1, varnames1, prog2, varnames2, axis, nsplit):

    for i in range(len(varnames1)):
        var1 = prog1.global_block().var(varnames1[i])
        var2 = prog2.global_block().var(varnames2[i])
        if var1.shape[axis] != (var2.shape[axis] // nsplit):
            return False

    return True


def initialization_check(mode, dist_context, dist_startup_prog,
                         serial_startup_prog, var_need_broadcast, process_mesh,
                         mp_parallel_axis, dp_parallel_axis):
    if 'mp' in mode:
        group_ranks = _get_comm_group(process_mesh.processes,
                                      process_mesh.topology, mp_parallel_axis,
                                      3)
        mp_ring_id = new_process_group(group_ranks).id
        broadcast_ops = [
            op for op in dist_startup_prog.global_block().ops if
            (op.type == "c_broadcast" and op.desc.attr("ring_id") == mp_ring_id)
        ]
        broadcast_varnames = sorted(
            [op.desc.output_arg_names()[0] for op in broadcast_ops])
        if broadcast_varnames != var_need_broadcast:
            return False

    if 'dp' in mode:
        group_ranks = _get_comm_group(process_mesh.processes,
                                      process_mesh.topology, dp_parallel_axis,
                                      3)
        dp_ring_id = new_process_group(group_ranks).id
        nparam = len(serial_startup_prog.all_parameters())
        nbroadcast_dp = len([
            op for op in dist_startup_prog.global_block().ops if
            (op.type == "c_broadcast" and op.desc.attr("ring_id") == dp_ring_id)
        ])
        if nparam != nbroadcast_dp:
            return False

    if "dp" in mode and 'mp' in mode:
        nbroadcast = len([
            op for op in dist_startup_prog.global_block().ops
            if op.type == "c_broadcast"
        ])
        if len(var_need_broadcast) + nbroadcast_dp != nbroadcast:
            return False

    return True


def get_input_var_dist_attr(op, main_program, dist_context):
    varname = op.desc.input_arg_names()
    var = main_program.global_block().var(varname[0])
    dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
    return dist_attr


def get_output_var_dist_attr(op, main_program, dist_context):
    varname = op.desc.output_arg_names()
    var = main_program.global_block().var(varname[0])
    dist_attr = dist_context.get_tensor_dist_attr_for_program(var)
    return dist_attr


def check_equal_var_dist_attr(serial_dist_attr, dist_attr):
    equal = True
    if serial_dist_attr.process_mesh != dist_attr.process_mesh or \
        serial_dist_attr.dims_mapping != dist_attr.dims_mapping:
        equal = False
    return equal


def check_equal_dist_op_attr(dist_context, dist_main_prog, serial_op, dist_ops,
                             dist_op_idx):
    equal = True
    # get serial op's process_mesh and impl_idx
    serial_op_dist_attr = dist_context.get_op_dist_attr_for_program(serial_op)
    serial_process_mesh = serial_op_dist_attr.process_mesh
    serial_impl_idx = serial_op_dist_attr.impl_idx

    # check dist_attr between serial op and dist op
    for i in dist_op_idx:
        op_dist_attr = dist_context.get_op_dist_attr_for_program(dist_ops[i])
        for in_varname in dist_ops[i].desc.input_arg_names():
            in_var = dist_main_prog.global_block().var(in_varname)
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                in_var)
            tensor_dims_mapping = tensor_dist_attr.dims_mapping
            in_var_dims_mapping = op_dist_attr.get_input_dims_mapping(
                in_varname)
            if tensor_dims_mapping != in_var_dims_mapping:
                equal = False
        for out_varname in dist_ops[i].desc.output_arg_names():
            out_var = dist_main_prog.global_block().var(out_varname)
            tensor_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                out_var)
            tensor_dims_mapping = tensor_dist_attr.dims_mapping
            out_var_dims_mapping = op_dist_attr.get_output_dims_mapping(
                out_varname)
            if tensor_dims_mapping != out_var_dims_mapping:
                equal = False
        dist_op_process_mesh = op_dist_attr.process_mesh
        dist_op_impl_idx = op_dist_attr.impl_idx
        if serial_op.desc.id() == dist_ops[i].desc.id() or \
            serial_process_mesh != dist_op_process_mesh or \
            serial_impl_idx != dist_op_impl_idx:
            equal = False

    return equal


def distributed_attr_check_for_dist_op(serial_main_prog, dist_main_prog,
                                       dist_context, serial_op_idx,
                                       dist_op_idx):

    equal = True
    serial_ops = serial_main_prog.global_block().ops
    dist_ops = dist_main_prog.global_block().ops

    for i in range(len(serial_op_idx)):
        serial_op = serial_ops[serial_op_idx[i]]
        dist_op_0 = dist_ops[dist_op_idx[i][0]]
        if dist_op_0.type == "c_identity":
            # serial op input's dist_attr
            serial_in_dist_attr = get_input_var_dist_attr(
                serial_op, serial_main_prog, dist_context)
            # c_identity output's(new var) dist_attr
            identity_out_dist_attr = get_output_var_dist_attr(
                dist_op_0, dist_main_prog, dist_context)
            # check var dist_attr
            equal = check_equal_var_dist_attr(serial_in_dist_attr,
                                              identity_out_dist_attr)
        else:
            # serial op output's dist_attr
            serial_out_dist_attr = get_output_var_dist_attr(
                serial_op, serial_main_prog, dist_context)
            # dist op output's(new var) dist_attr
            out_dist_attr = get_output_var_dist_attr(dist_op_0, dist_main_prog,
                                                     dist_context)
            # check var dist_attr
            equal = check_equal_var_dist_attr(serial_out_dist_attr,
                                              out_dist_attr)

        # check op's dist_attr
        equal = check_equal_dist_op_attr(dist_context, dist_main_prog,
                                         serial_op, dist_ops, dist_op_idx[i])

    return equal


def distributed_attr_check_for_program(dist_main_prog, dist_context):
    have_dist_attr = True
    for block in dist_main_prog.blocks:
        for tensor in block.vars.values():
            var_dist_attr = dist_context.get_tensor_dist_attr_for_program(
                tensor)
            if var_dist_attr is None:
                have_dist_attr = False

        for op in block.ops:
            op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
            if op_dist_attr is None:
                have_dist_attr = False

    return have_dist_attr


class MLPLayer(nn.Layer):

    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None

        self.linear0 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        if _global_parallel_strategy in ["mp", "dp_mp"]:
            auto.shard_tensor(self.linear0.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])
            auto.shard_tensor(self.linear1.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=["mp", None])
        else:
            auto.shard_tensor(self.linear0.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, None])
            auto.shard_tensor(self.linear1.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, None])

        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)

        return out


def mlp_pretrain_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name="input",
                            shape=[batch_size, sequence_len, hidden_size],
                            dtype='float32')

        if _global_parallel_strategy in ["dp", "dp_mp"]:
            auto.shard_tensor(input,
                              process_mesh=_global_process_mesh,
                              shard_spec=["dp", None, None])

        mlp = MLPLayer(hidden_size=hidden_size,
                       intermediate_size=4 * hidden_size,
                       dropout_ratio=0.1,
                       initializer_range=0.02)
        out = mlp(input)
    return train_program, start_program


class TestMLPAutoPartitioner(unittest.TestCase):

    def test_mlp_dp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3],
                                                dim_names=["dp"])

        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            mlp_pretrain_forward)

        # parameter should not be partitioned
        self.assertTrue(
            is_all_parameters_shape_equal(serial_main_prog, dist_main_prog))
        self.assertTrue(
            is_all_parameters_shape_equal(serial_startup_prog,
                                          dist_startup_prog))

        # op in main prog should be the same
        serial_ops = serial_main_prog.global_block().ops
        dist_ops = dist_main_prog.global_block().ops
        serial_ops = [op.type for op in serial_ops]
        dist_ops = [op.type for op in dist_ops]
        self.assertTrue(serial_ops == dist_ops)

        # parameter initialization
        var_need_broadcast = []
        self.assertTrue(
            initialization_check(_global_parallel_strategy,
                                 dist_context,
                                 dist_startup_prog,
                                 serial_startup_prog,
                                 var_need_broadcast,
                                 _global_process_mesh,
                                 mp_parallel_axis=None,
                                 dp_parallel_axis=0))

    def test_mlp_mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3],
                                                dim_names=["mp"])
        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            mlp_pretrain_forward)

        # param should be partition
        nrank = 4
        # col parallel
        weights = ['linear_0.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 1, nrank))
        weights = ['linear_0.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        # row parallel
        weights = ['linear_1.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        weights = ['linear_1.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, 1))

        # row and col allreduce
        dist_ops = dist_main_prog.global_block().ops
        dist_ops = [op.type for op in dist_ops]
        ref_ops = [
            'layer_norm', 'c_identity', 'matmul_v2', 'elementwise_add', 'gelu',
            'matmul_v2', 'c_allreduce_sum', 'elementwise_add', 'dropout'
        ]
        self.assertTrue(dist_ops == ref_ops)

        # parameter initialization
        var_need_broadcast = sorted(
            ['layer_norm_0.b_0', 'layer_norm_0.w_0', 'linear_1.b_0'])
        self.assertTrue(
            initialization_check(_global_parallel_strategy,
                                 dist_context,
                                 dist_startup_prog,
                                 serial_startup_prog,
                                 var_need_broadcast,
                                 _global_process_mesh,
                                 mp_parallel_axis=0,
                                 dp_parallel_axis=None))

        # check var and op all have dist_attr in dist_main_program
        self.assertTrue(
            distributed_attr_check_for_program(dist_main_prog, dist_context))
        # check distribured attr for dist op
        serial_op_idx = [1, 4]
        dist_op_idx = [[1, 2], [5, 6]]
        self.assertTrue(
            distributed_attr_check_for_dist_op(serial_main_prog, dist_main_prog,
                                               dist_context, serial_op_idx,
                                               dist_op_idx))

    def test_mlp_dp_mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp_mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3],
                                                      [4, 5, 6, 7]],
                                                dim_names=["dp", "mp"])
        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            mlp_pretrain_forward)

        # param should be partition
        nrank = 4
        # col parallel
        weights = ['linear_0.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 1, nrank))
        weights = ['linear_0.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        # row parallel
        weights = ['linear_1.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        weights = ['linear_1.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, 1))

        # row and col allreduce
        dist_ops = dist_main_prog.global_block().ops
        dist_ops = [op.type for op in dist_ops]
        ref_ops = [
            'layer_norm', 'c_identity', 'matmul_v2', 'elementwise_add', 'gelu',
            'matmul_v2', 'c_allreduce_sum', 'elementwise_add', 'dropout'
        ]
        self.assertTrue(dist_ops == ref_ops)

        # parameter initialization
        var_need_broadcast = sorted(
            ['layer_norm_0.b_0', 'layer_norm_0.w_0', 'linear_1.b_0'])
        self.assertTrue(
            initialization_check(_global_parallel_strategy,
                                 dist_context,
                                 dist_startup_prog,
                                 serial_startup_prog,
                                 var_need_broadcast,
                                 _global_process_mesh,
                                 mp_parallel_axis=1,
                                 dp_parallel_axis=0))

        # check var and op all have dist_attr in dist_main_program
        self.assertTrue(
            distributed_attr_check_for_program(dist_main_prog, dist_context))
        # check distribured attr for dist op
        serial_op_idx = [1, 4]
        dist_op_idx = [[1, 2], [5, 6]]
        self.assertTrue(
            distributed_attr_check_for_dist_op(serial_main_prog, dist_main_prog,
                                               dist_context, serial_op_idx,
                                               dist_op_idx))


class AttentionLayer(nn.Layer):

    def __init__(self,
                 hidden_size=1024,
                 sequence_len=512,
                 intermediate_size=4 * 1024,
                 num_heads=16,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len
        self.embed_dim = self.hidden_size
        self.kdim = self.embed_dim
        self.vdim = self.embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.dropout_ratio = dropout_ratio
        self.initializer_range = initializer_range
        self.training = True
        self.attn_mask = None
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None

        self.q_proj = nn.Linear(self.embed_dim,
                                self.embed_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.k_proj = nn.Linear(self.kdim,
                                self.embed_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.v_proj = nn.Linear(self.vdim,
                                self.embed_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.out_proj = nn.Linear(self.embed_dim,
                                  self.embed_dim,
                                  weight_attr,
                                  bias_attr=bias_attr)

    def forward(self, input):
        if _global_parallel_strategy in ["dp", "dp_mp"]:
            auto.shard_tensor(input,
                              process_mesh=_global_process_mesh,
                              shard_spec=["dp", None, None])

        q = self.q_proj(input)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        k = self.k_proj(input)
        v = self.v_proj(input)

        if _global_parallel_strategy in ["mp", "dp_mp"]:
            auto.shard_tensor(self.q_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])
            auto.shard_tensor(self.k_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])
            auto.shard_tensor(self.v_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])

        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        # scale dot product attention
        product = layers.matmul(x=q,
                                y=k,
                                transpose_y=True,
                                alpha=self.head_dim**-0.5)

        if self.attn_mask is not None:
            product = product + self.attn_mask

        weights = F.softmax(product)

        if self.dropout_ratio:
            weights = F.dropout(weights,
                                self.dropout_ratio,
                                training=self.training,
                                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        if _global_parallel_strategy in ["mp", "dp_mp"]:
            auto.shard_tensor(self.out_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=["mp", None])

        return out


def attn_pretrain_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name="query",
                            shape=[batch_size, sequence_len, hidden_size],
                            dtype='float32')
        attn = AttentionLayer(hidden_size=hidden_size,
                              sequence_len=sequence_len,
                              intermediate_size=4 * hidden_size,
                              num_heads=16,
                              dropout_ratio=0.1,
                              initializer_range=0.02)
        out = attn(input)

    return train_program, start_program


class TestAttentionAutoPartitioner(unittest.TestCase):

    def test_attn_dp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3],
                                                dim_names=["dp"])

        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            attn_pretrain_forward)
        # parameter should not be partitioned
        self.assertTrue(
            is_all_parameters_shape_equal(serial_main_prog, dist_main_prog))
        self.assertTrue(
            is_all_parameters_shape_equal(serial_startup_prog,
                                          dist_startup_prog))

        # op in main prog should be the same
        serial_ops = serial_main_prog.global_block().ops
        dist_ops = dist_main_prog.global_block().ops
        serial_ops = [op.type for op in serial_ops]
        dist_ops = [op.type for op in dist_ops]
        self.assertTrue(serial_ops == dist_ops)

        # parameter initialization
        var_need_broadcast = []
        self.assertTrue(
            initialization_check(_global_parallel_strategy,
                                 dist_context,
                                 dist_startup_prog,
                                 serial_startup_prog,
                                 var_need_broadcast,
                                 _global_process_mesh,
                                 mp_parallel_axis=None,
                                 dp_parallel_axis=0))

    def test_attn_mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3],
                                                dim_names=["mp"])

        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            attn_pretrain_forward)

        # param should be partition
        nrank = 4
        # col parallel
        weights = ['linear_0.w_0', 'linear_1.w_0', 'linear_2.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 1, nrank))
        weights = ['linear_0.b_0', 'linear_1.b_0', 'linear_2.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        # row parallel
        weights = ['linear_3.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        weights = ['linear_3.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, 1))

        # row and col allreduce
        dist_ops = dist_main_prog.global_block().ops
        dist_ops = [op.type for op in dist_ops]
        ref_ops = [
            'c_identity', 'matmul_v2', 'elementwise_add', 'reshape2',
            'transpose2', 'c_identity', 'matmul_v2', 'elementwise_add',
            'c_identity', 'matmul_v2', 'elementwise_add', 'reshape2',
            'transpose2', 'reshape2', 'transpose2', 'matmul', 'softmax',
            'dropout', 'matmul_v2', 'transpose2', 'reshape2', 'matmul_v2',
            'c_allreduce_sum', 'elementwise_add'
        ]
        self.assertTrue(dist_ops == ref_ops)

        # parameter initialization
        var_need_broadcast = ['linear_3.b_0']
        self.assertTrue(
            initialization_check(_global_parallel_strategy,
                                 dist_context,
                                 dist_startup_prog,
                                 serial_startup_prog,
                                 var_need_broadcast,
                                 _global_process_mesh,
                                 mp_parallel_axis=0,
                                 dp_parallel_axis=None))

        # check var and op all have dist_attr in dist_main_program
        self.assertTrue(
            distributed_attr_check_for_program(dist_main_prog, dist_context))
        # check distribured attr for dist op
        serial_op_idx = [0, 4, 6, 18]
        dist_op_idx = [[0, 1], [5, 6], [8, 9], [21, 22]]
        self.assertTrue(
            distributed_attr_check_for_dist_op(serial_main_prog, dist_main_prog,
                                               dist_context, serial_op_idx,
                                               dist_op_idx))

    def test_attn_dp_mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp_mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3],
                                                      [4, 5, 6, 7]],
                                                dim_names=["dp", "mp"])

        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            attn_pretrain_forward)

        # param should be partition
        nrank = 4
        # col parallel
        weights = ['linear_0.w_0', 'linear_1.w_0', 'linear_2.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 1, nrank))
        weights = ['linear_0.b_0', 'linear_1.b_0', 'linear_2.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        # row parallel
        weights = ['linear_3.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        weights = ['linear_3.b_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, 1))

        # row and col allreduce
        dist_ops = dist_main_prog.global_block().ops
        dist_ops = [op.type for op in dist_ops]
        ref_ops = [
            'c_identity', 'matmul_v2', 'elementwise_add', 'reshape2',
            'transpose2', 'c_identity', 'matmul_v2', 'elementwise_add',
            'c_identity', 'matmul_v2', 'elementwise_add', 'reshape2',
            'transpose2', 'reshape2', 'transpose2', 'matmul', 'softmax',
            'dropout', 'matmul_v2', 'transpose2', 'reshape2', 'matmul_v2',
            'c_allreduce_sum', 'elementwise_add'
        ]
        self.assertTrue(dist_ops == ref_ops)

        # parameter initialization
        var_need_broadcast = ['linear_3.b_0']
        self.assertTrue(
            initialization_check(_global_parallel_strategy,
                                 dist_context,
                                 dist_startup_prog,
                                 serial_startup_prog,
                                 var_need_broadcast,
                                 _global_process_mesh,
                                 mp_parallel_axis=1,
                                 dp_parallel_axis=0))

        # check var and op all have dist_attr in dist_main_program
        self.assertTrue(
            distributed_attr_check_for_program(dist_main_prog, dist_context))
        # check distribured attr for dist op
        serial_op_idx = [0, 4, 6, 18]
        dist_op_idx = [[0, 1], [5, 6], [8, 9], [21, 22]]
        self.assertTrue(
            distributed_attr_check_for_dist_op(serial_main_prog, dist_main_prog,
                                               dist_context, serial_op_idx,
                                               dist_op_idx))


class DecoderLayer(nn.Layer):

    def __init__(self,
                 vocab_size=32768,
                 hidden_size=1024,
                 sequence_len=512,
                 max_position_embeddings=512,
                 intermediate_size=4 * 1024,
                 num_heads=16,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(DecoderLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.sequence_len = sequence_len
        self.embed_dim = self.hidden_size
        self.kdim = self.embed_dim
        self.vdim = self.embed_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio
        self.initializer_range = initializer_range
        self.training = True
        self.attn_mask = None

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.word_embeddings = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            weight_attr=paddle.ParamAttr(name="word_embeddings",
                                         initializer=nn.initializer.Normal(
                                             mean=0.0,
                                             std=self.initializer_range)))
        self.position_embeddings = nn.Embedding(
            self.max_position_embeddings,
            self.hidden_size,
            weight_attr=paddle.ParamAttr(name="pos_embeddings",
                                         initializer=nn.initializer.Normal(
                                             mean=0.0,
                                             std=self.initializer_range)))

        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=self.initializer_range))
        bias_attr = None
        self.q_proj = nn.Linear(self.embed_dim,
                                self.embed_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.k_proj = nn.Linear(self.kdim,
                                self.embed_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.v_proj = nn.Linear(self.vdim,
                                self.embed_dim,
                                weight_attr,
                                bias_attr=bias_attr)
        self.out_proj = nn.Linear(self.embed_dim,
                                  self.embed_dim,
                                  weight_attr,
                                  bias_attr=bias_attr)

        intermediate_size = 4 * self.hidden_size
        d_model = self.hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=self.initializer_range))
        bias_attr = None
        self.linear0 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attr,
                                 bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(self.dropout_ratio)
        self.dropout2 = nn.Dropout(self.dropout_ratio, mode="upscale_in_train")
        self.dropout3 = nn.Dropout(self.dropout_ratio, mode="upscale_in_train")

    def forward(self, input_ids, position_ids):
        if _global_parallel_strategy in ["dp", "dp_mp"]:
            auto.shard_tensor(input_ids,
                              process_mesh=_global_process_mesh,
                              shard_spec=["dp", None])

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if _global_parallel_strategy in ["mp", "dp_mp"]:
            auto.shard_tensor(self.word_embeddings.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=["mp", None])

        embeddings = input_embeddings + position_embeddings
        embeddings = self.dropout1(embeddings)

        # Pre-norm
        target = self.norm(embeddings)

        # The following is the attention part
        q = self.q_proj(target)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        k = self.k_proj(target)
        v = self.v_proj(target)

        if _global_parallel_strategy in ["mp", "dp_mp"]:
            auto.shard_tensor(self.q_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])
            auto.shard_tensor(self.k_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])
            auto.shard_tensor(self.v_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])

        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])

        # scale dot product attention
        product = layers.matmul(x=q,
                                y=k,
                                transpose_y=True,
                                alpha=self.head_dim**-0.5)

        if self.attn_mask is not None:
            product = product + self.attn_mask

        weights = F.softmax(product)

        if self.dropout_ratio:
            weights = F.dropout(weights,
                                self.dropout_ratio,
                                training=self.training,
                                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        if _global_parallel_strategy in ["mp", "dp_mp"]:
            auto.shard_tensor(self.out_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=["mp", None])
        else:
            auto.shard_tensor(self.out_proj.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, None])

        # Add residual
        residual = embeddings + self.dropout2(out)

        # Pre-norm
        out0 = self.norm(residual)

        # The following is the MLP part
        out1 = self.linear0(out0)
        out2 = F.gelu(out1, approximate=True)
        out3 = self.linear1(out2)

        if _global_parallel_strategy in ["mp", "dp_mp"]:
            auto.shard_tensor(self.linear0.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=[None, "mp"])
            auto.shard_tensor(self.linear1.weight,
                              process_mesh=_global_process_mesh,
                              shard_spec=["mp", None])

        # Add residual
        final = residual + self.dropout3(out3)
        return final


def decoder_pretrain_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input_ids = static.data(name="input_ids",
                                shape=[batch_size, sequence_len],
                                dtype='int64')
        position_ids = static.data(name="position_ids",
                                   shape=[batch_size, sequence_len],
                                   dtype='int64')
        decoder = DecoderLayer(vocab_size=32768,
                               hidden_size=hidden_size,
                               sequence_len=sequence_len,
                               max_position_embeddings=512,
                               intermediate_size=4 * hidden_size,
                               num_heads=16,
                               dropout_ratio=0.1,
                               initializer_range=0.02)
        out = decoder(input_ids, position_ids)

    return train_program, start_program


class TestDecoderLayerPartitioner(unittest.TestCase):

    def test_decoder_dp_mp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp_mp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3],
                                                      [4, 5, 6, 7]],
                                                dim_names=["dp", "mp"])
        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            decoder_pretrain_forward)

        # param should be partition
        nrank = 4
        # col parallel
        weights = [
            'linear_0.w_0', 'linear_1.w_0', 'linear_2.w_0', 'linear_4.w_0'
        ]
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 1, nrank))
        weights = [
            'linear_0.b_0', 'linear_1.b_0', 'linear_2.b_0', 'linear_4.b_0'
        ]
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        # row parallel
        weights = ['word_embeddings', 'linear_3.w_0', 'linear_5.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        weights = [
            'linear_3.b_0', 'pos_embeddings', 'layer_norm_0.b_0',
            'layer_norm_0.w_0', 'linear_5.b_0'
        ]
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, 1))

        # row and col allreduce
        dist_ops = dist_main_prog.global_block().ops
        dist_ops = [op.type for op in dist_ops]
        ref_ops = [
            'c_embedding', 'c_allreduce_sum', 'lookup_table_v2',
            'elementwise_add', 'dropout', 'layer_norm', 'c_identity',
            'matmul_v2', 'elementwise_add', 'reshape2', 'transpose2',
            'c_identity', 'matmul_v2', 'elementwise_add', 'c_identity',
            'matmul_v2', 'elementwise_add', 'reshape2', 'transpose2',
            'reshape2', 'transpose2', 'matmul', 'softmax', 'dropout',
            'matmul_v2', 'transpose2', 'reshape2', 'matmul_v2',
            'c_allreduce_sum', 'elementwise_add', 'dropout', 'elementwise_add',
            'layer_norm', 'c_identity', 'matmul_v2', 'elementwise_add', 'gelu',
            'matmul_v2', 'c_allreduce_sum', 'elementwise_add', 'dropout',
            'elementwise_add'
        ]
        self.assertTrue(dist_ops == ref_ops)

        # parameter initialization
        var_need_broadcast = sorted([
            'linear_3.b_0', 'pos_embeddings', 'layer_norm_0.b_0',
            'layer_norm_0.w_0', 'linear_5.b_0'
        ])
        self.assertTrue(
            initialization_check(_global_parallel_strategy,
                                 dist_context,
                                 dist_startup_prog,
                                 serial_startup_prog,
                                 var_need_broadcast,
                                 _global_process_mesh,
                                 mp_parallel_axis=1,
                                 dp_parallel_axis=0))

        # check var and op all have dist_attr in dist_main_program
        self.assertTrue(
            distributed_attr_check_for_program(dist_main_prog, dist_context))
        # check distribured attr
        serial_op_idx = [0, 5, 9, 11, 23, 28, 31]
        dist_op_idx = [[0, 1], [6, 7], [11, 12], [14, 15], [27, 28], [33, 34],
                       [37, 38]]
        self.assertTrue(
            distributed_attr_check_for_dist_op(serial_main_prog, dist_main_prog,
                                               dist_context, serial_op_idx,
                                               dist_op_idx))

    def test_decoder_noparallel(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "None"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3],
                                                      [4, 5, 6, 7]],
                                                dim_names=["x", "y"])
        serial_main_prog, serial_startup_prog, dist_main_prog, dist_startup_prog, dist_context = get_programs(
            decoder_pretrain_forward)

        # param should be partition
        nrank = 1
        # col parallel
        weights = [
            'linear_0.w_0', 'linear_1.w_0', 'linear_2.w_0', 'linear_4.w_0'
        ]
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 1, nrank))
        weights = [
            'linear_0.b_0', 'linear_1.b_0', 'linear_2.b_0', 'linear_4.b_0'
        ]
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        # row parallel
        weights = ['word_embeddings', 'linear_3.w_0', 'linear_5.w_0']
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, nrank))
        weights = [
            'linear_3.b_0', 'pos_embeddings', 'layer_norm_0.b_0',
            'layer_norm_0.w_0', 'linear_5.b_0'
        ]
        self.assertTrue(
            check_tensor_split(dist_main_prog, weights, serial_main_prog,
                               weights, 0, 1))

        # row and col allreduce
        dist_ops = dist_main_prog.global_block().ops
        dist_ops = [op.type for op in dist_ops]
        ref_ops = [
            'lookup_table_v2', 'lookup_table_v2', 'elementwise_add', 'dropout',
            'layer_norm', 'matmul_v2', 'elementwise_add', 'reshape2',
            'transpose2', 'matmul_v2', 'elementwise_add', 'matmul_v2',
            'elementwise_add', 'reshape2', 'transpose2', 'reshape2',
            'transpose2', 'matmul', 'softmax', 'dropout', 'matmul_v2',
            'transpose2', 'reshape2', 'matmul_v2', 'elementwise_add', 'dropout',
            'elementwise_add', 'layer_norm', 'matmul_v2', 'elementwise_add',
            'gelu', 'matmul_v2', 'elementwise_add', 'dropout', 'elementwise_add'
        ]
        self.assertTrue(dist_ops == ref_ops)
        dist_ops = dist_startup_prog.global_block().ops
        dist_ops = [op.type for op in dist_ops]
        ref_ops = [
            'gaussian_random', 'gaussian_random', 'gaussian_random',
            'fill_constant', 'gaussian_random', 'fill_constant',
            'gaussian_random', 'fill_constant', 'gaussian_random',
            'fill_constant', 'gaussian_random', 'fill_constant',
            'gaussian_random', 'fill_constant', 'fill_constant',
            'fill_constant', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast', 'c_broadcast', 'c_broadcast', 'c_broadcast',
            'c_broadcast'
        ]
        self.assertTrue(dist_ops == ref_ops)


if __name__ == "__main__":
    unittest.main()
