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

import paddle
from paddle import _C_ops
from paddle.fluid.framework import _non_static_mode
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype
from .reduce import ReduceOp
from .utils import _check_single_tensor
from .reduce import _get_reduce_op
from .group import _get_default_group

__all__ = ["reduce_scatter", "_check_tensor_list"]


def _check_tensor_list(tensor_list, tensor_name):
    if not isinstance(tensor_list, list) or \
        not all(isinstance(t, (core.eager.Tensor, paddle.Tensor)) for t in tensor_list):
        raise RuntimeError("Invalid function argument. Expected parameter {}"
                           "to be of type paddle.Tensor".format(tensor_name))


def reduce_scatter(tensor,
                   tensor_list,
                   op=ReduceOp.SUM,
                   group=None,
                   use_calc_stream=True):
    """
    Reduces, then scatters a list of tensors to all processes in a group

    Args:
        tensor (Tensor): Output tensor.
        tensor_list (list[Tensor]): List of tensors to reduce and scatter.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used. Default: ReduceOp.SUM.
        group (Group, optional): The group instance return by new_group or None for global 
            default group. Default: None.
        use_calc_stream (bool, optional): Whether this op should be an async op.

    Returns:
        Async task handle, if use_calc_stream is set to False.
        None, if use_calc_stream or if not part of the group.
    
    Warning:    
        This API only supports the dygraph mode.


    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            if rank == 0:
                t1 = paddle.to_tensor([0, 1])
                t2 = paddle.to_tensor([2, 3])
            else:
                t1 = paddle.to_tensor([4, 5])
                t2 = paddle.to_tensor([6, 7])

            tensor_list = [t1, t2]

            output = paddle.empty(shape=[2], dtype=tensor_list[0].dtype)
            dist.reduce_scatter(output, tensor_list)

            print(output)
            # [4, 6]     # Rank-0
            # [8, 10]     # Rank-1

    """
    _check_single_tensor(tensor, "tensor")
    _check_tensor_list(tensor_list, "tensor_list")

    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        op_type = _get_reduce_op(op, "reduce_scatter")
        group = _get_default_group() if group is None else group

        temp = paddle.concat(tensor_list, axis=0)
        task = group.process_group._reduce_scatter_base(tensor, temp, op_type)
        if use_calc_stream:
            task.wait()
            return None
        else:
            return task
    else:
        raise RuntimeError("Don't support static graph mode currently.")
