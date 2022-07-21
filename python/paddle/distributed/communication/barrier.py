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
from .group import _get_default_group

__all__ = ["barrier"]

def _dygraph_barrier(group=None):
    group = _get_default_group() if group is None else group
    task = group.process_group.barrier()
    task.wait()

def _static_barrier(group=None):
    ring_id = 0 if group is None else group.id

    temp = fill_constant([1], dtype="int32", value="1")
    if _non_static_mode():
        return _C_ops.barrier(temp, temp, 'ring_id', ring_id)

    op_type = 'barrier'

    if not isinstance(ring_id, int):
        raise ValueError("The type of 'group' for barrier must be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [temp]},
                     outputs={'Out': [temp]},
                     attrs={'ring_id': ring_id})

def barrier(group=None):
    """

    Barrier among all participators in the group.

    Args:
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            paddle.distributed.barrier()
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        _dygraph_barrier(group)
        return

    _static_barrier(group)
