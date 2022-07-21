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
import contextlib
from paddle import _C_ops
from paddle.fluid.framework import in_dygraph_mode
from .p2p import _check_p2p_op_list
from .group import _get_default_group, _group_map_backend

__all__ = ["batch_isend_irecv", "_with_batch_p2p_guard"]

@contextlib.contextmanager
def _with_batch_p2p_guard(backend):
    if backend == "nccl":
        core.ProcessGroupNCCL.group_start()
    try:
        yield
    finally:
        if backend == "nccl":
            core.ProcessGroupNCCL.group_end()

def batch_isend_irecv(p2p_op_list):
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the point-to-point operations in ``p2p_op_list`` and return the 
    corresponding tasks. NCCL are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``paddle.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed tasks returned by calling the corresponding
        op in the op_list. 

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

            send_t = paddle.arange(2) + rank
            # paddle.tensor([0, 1])  # Rank-0
            # paddle.tensor([1, 2])  # Rank-1

            recv_t = paddle.empty(shape=[2], dtype=send_t.dtype)

            send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)
            recv_op = dist.P2POp(dist.irecv, recv_t, (rank - 1 + world_size) % world_size)

            tasks = dist.batch_isend_irecv([send_op, recv_op])

            for task in tasks:
                task.wait()
            
            print(recv_t)
            # paddle.tensor([1, 2])     # Rank-0
            # paddle.tensor([0, 1])     # Rank-1
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        backend = _group_map_backend[group]
        tasks = []
        with _with_batch_p2p_guard(backend):
            for p2p_op in p2p_op_list:
                op = p2p_op.op
                tensor = p2p_op.tensor
                peer = p2p_op.peer
                comm_group = p2p_op.group
                task = op(tensor, peer, comm_group)
                if task is not None:
                    tasks.append(task)
        return tasks
    else:
        raise RuntimeError("Don't support static graph mode currently.")
