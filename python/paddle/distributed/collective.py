#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import _C_ops
from .communication.all_gather import *
from .communication.alltoall import *
from .communication.batch_isend_irecv import *
from .communication.group import *
from .communication.p2p import *
from .communication.reduce import *
from .communication.scatter import *
from .communication.split import *
from .communication.wait import *
from .communication.all_reduce import *
from .communication.barrier import *
from .communication.broadcast import *
from .communication.recv import *
from .communication.reduce_scatter import *
from .communication.send import *
from .communication.utils import *
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import in_dygraph_mode
from ..fluid.framework import _non_static_mode
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.layers.tensor import fill_constant
from ..fluid.layers import utils
from ..fluid.dygraph import layers

__all__ = [
    "Group",
    "new_group",
    "get_group",
    "is_initialized",
    "destroy_process_group",
    "scatter",
    "barrier",
    "broadcast",
    "recv",
    "irecv",
    "reduce_scatter",
    "send",
    "isend",
    "all_reduce",
    "all_gather",
    "alltoall",
    "alltoall_single",
    "batch_isend_irecv",
    "split",
    "wait",
    "P2POp",
    "reduce",
    "ReduceOp",
    "_sync_calc_stream",
    "_sync_comm_stream",
    "_mp_allreduce",
    "_c_lookup_table",
    "_parallel_embedding",
    "_c_split",
    "_c_identity",
    "_linear",
    "_parallel_linear",
    "_check_p2p_op_list",
    "_get_global_env",
    "_set_default_backend",
    "_set_default_store",
    "_get_group_map",
    "_get_global_group",
    "_get_group_map_by_name",
    "_get_default_group",
    "_set_group_map",
    "_set_group_map_by_name",
    "_set_group_map_backend",
    "_new_ring_id",
    "_new_process_group_impl",
    "_set_custom_gid",
    "_c_concat",
    "_Linear",
    "_c_softmax_with_cross_entropy",
    "_reduce_scatter_base",
    "_with_batch_p2p_guard",
    "_get_reduce_op",
    "_get_group_rank",
    "_check_single_tensor",
    "_check_tensor_list",
    "_set_var_distributed",
]


def _c_concat(tensor, group=None):
    """
    Return allgather of the tensor, mainly used with model parallel.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.

    Returns:
        Tensor.
    """
    if group is not None and not group.is_member():
        return

    group = _get_default_group() if group is None else group
    ring_id = group.id

    global_rank = _get_global_env().rank
    rank = group.rank
    nranks = group.nranks

    if _non_static_mode():
        return _C_ops.c_concat(tensor, 'ring_id', ring_id, 'use_calc_stream',
                               True, 'rank', rank, 'nranks', nranks,
                               'use_model_parallel', True)

    op_type = 'c_concat'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        '_c_concat')

    helper.append_op(type=op_type,
                     inputs={'X': tensor},
                     outputs={'Out': out},
                     attrs={
                         'ring_id': ring_id,
                         'use_calc_stream': True,
                         'use_model_parallel': True,
                         'nranks': nranks,
                         'rank': rank
                     })
    return out


class _Linear(layers.Layer):
    """
    Linear
    """

    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(_Linear, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(shape=[in_features, out_features],
                                            attr=self._weight_attr,
                                            dtype=self._dtype,
                                            is_bias=False)
        self.bias = self.create_parameter(shape=[out_features],
                                          attr=self._bias_attr,
                                          dtype=self._dtype,
                                          is_bias=True)
        self.name = name

    def forward(self, input):
        out = _linear(x=input,
                      weight=self.weight,
                      bias=self.bias,
                      name=self.name)
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}'.format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str)


def _c_softmax_with_cross_entropy(logits,
                                  label,
                                  group=None,
                                  return_softmax=False):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    global_rank = _get_global_env().rank
    rank = global_rank if group is None else group.get_group_rank(global_rank)
    nranks = _get_global_env().world_size if group is None else group.nranks

    input_dims = len(list(logits.shape))
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            'Expected nput_dims - 1 = label_dims or input_dims == label_dims\
             (got nput_dims{}, label_dims{})'.format(input_dims, label_dims))
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=-1)

    if _non_static_mode():
        softmax, loss = _C_ops.c_softmax_with_cross_entropy(
            logits, label, 'ring_id', ring_id, 'rank', rank, 'nranks', nranks)
        if not return_softmax:
            return loss
        else:
            return loss, softmax

    attrs = {
        'ring_id': ring_id,
        'rank': rank,
        'nranks': nranks,
    }
    helper = LayerHelper('c_softmax_with_cross_entropy', **locals())
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
    helper.append_op(type='c_softmax_with_cross_entropy',
                     inputs={
                         'Logits': logits,
                         'Label': label
                     },
                     outputs={
                         'Softmax': softmax,
                         'Loss': loss
                     },
                     attrs=attrs)

    if return_softmax:
        return loss, softmax

    return loss


def _reduce_scatter_base(output,
                         input,
                         op=ReduceOp.SUM,
                         group=None,
                         use_calc_stream=True):
    """
    Reduces, then scatters a flattened tensor to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input (Tensor): Input tensor that is of size output tensor size times world size
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used. Default: ReduceOp.SUM.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        use_calc_stream (bool, optional): Wether to use calculation stream (True) or communication stream (False).
            Default to True.
    Returns:
        Async task handle, if use_calc_stream is set to False.
        None, if use_calc_stream or if not part of the group.

    Examples:
        .. code-block:: python

            # required: distributed

            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            input = paddle.arange(4) + rank
            # [0, 1, 2, 3]  # Rank-0
            # [1, 2, 3, 4]  # Rank-1

            output = paddle.empty(shape=[2], dtype=input.dtype)
            paddle.distributed.collective._reduce_scatter_base(output, input)
            print(output)
            # [1, 3]     # Rank-0
            # [5, 7]     # Rank-1

    """
    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")

    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        op_type = _get_reduce_op(op, "_reduce_scatter_base")
        group = _get_default_group() if group is None else group
        task = group.process_group._reduce_scatter_base(output, input, op_type)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task
    else:
        raise RuntimeError("Don't support static graph mode currently.")
