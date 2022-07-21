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
from .group import _get_default_group
from .utils import _check_single_tensor, _get_group_rank

__all__ = ["send", "isend"]


def _dygraph_send(tensor, dst=0, group=None, use_calc_stream=True):
    group = _get_default_group() if group is None else group
    task = group.process_group.send(tensor, dst)
    if use_calc_stream:
        task.wait()
        return None
    else:
        return task


def _static_send(tensor, dst=0, group=None, use_calc_stream=True):
    ring_id = 0 if group is None else group.id

    if _non_static_mode():
        return _C_ops.send_v2(tensor, 'use_calc_stream', use_calc_stream,
                              'ring_id', ring_id, 'peer', dst)
    op_type = 'send_v2'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'send')

    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [tensor]},
                     attrs={
                         'ring_id': ring_id,
                         'peer': dst,
                         'use_calc_stream': use_calc_stream,
                     })


def send(tensor, dst=0, group=None, use_calc_stream=True):
    """
    Send a tensor to the receiver.

    Args:
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
        dst (int): The destination rank id.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculate stream or communication stream. Default: True.
    
    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            from paddle.distributed import init_parallel_env

            init_parallel_env()
            if paddle.distributed.ParallelEnv().rank == 0:
                data = paddle.to_tensor([7, 8, 9])
                paddle.distributed.send(data, dst=1)
            else:
                data = paddle.to_tensor([1,2,3])
                paddle.distributed.recv(data, src=0)
            out = data.numpy()
    """
    if group is not None and not group.is_member():
        return
    dst = _get_group_rank(dst, group)
    if in_dygraph_mode():
        return _dygraph_send(tensor, dst, group, use_calc_stream)

    _static_send(tensor, dst, group, use_calc_stream)


def isend(tensor, dst, group=None):
    """
    Sends a tensor asynchronously

    Args:
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
        dst (int): The destination rank.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
    
    Returns:
        A distributed task object.

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
                data = paddle.to_tensor([7, 8, 9])
                task = paddle.distributed.isend(data, dst=1)
            else:
                data = paddle.to_tensor([1, 2, 3])
                task = paddle.distributed.irecv(data, src=0)

            task.wait()

            print(data)
            # paddle.tensor([7, 8, 9])     # Rank-0
            # paddle.tensor([7, 8, 9])     # Rank-1

    """
    _check_single_tensor(tensor, "tensor")
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        group_dst_rank = group.get_group_rank(dst)
        assert group_dst_rank >= 0, ("dst rank out of group, need global rank")
        return group.process_group.send(tensor, group_dst_rank)
    else:
        raise RuntimeError("Don't support static graph mode currently.")
