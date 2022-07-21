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
from .group import _group_map_backend, _get_default_group
from .utils import _check_single_tensor

__all__ = ["P2POp", "_check_p2p_op_list"]

class P2POp(object):
    """
    A class that makes point-to-point operations for "batch_isend_irecv".

    This class creates the type of P2P operation, communication buffer, peer rank,
    Group. Instances of this class will be passed to
    ``paddle.distributed.batch_isend_irecv`` for point-to-point communication.

    Args:
        op (callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``paddle.distributed.isend`` or ``paddle.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int): The destination or source rank.
        group (Group, optional): The group instance return by new_group or None for global 
            default group. Default: None.

    """

    def __init__(self, op, tensor, peer, group=None):
        if op not in [isend, irecv]:
            raise RuntimeError("Invalid ``op`` function. Expected ``op`` "
                               "to be of type ``paddle.distributed.isend`` or "
                               "``paddle.distributed.irecv``.")
        _check_single_tensor(tensor, "tensor")

        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = _get_default_group() if group is None else group

def _check_p2p_op_list(p2p_op_list):
    """
    Helper to check that the ``p2p_op_list`` is a list of P2POp instances and
    all ops use the same backend.
    """
    if not isinstance(p2p_op_list, list) or not all(
            isinstance(p2p_op, P2POp) for p2p_op in p2p_op_list):
        raise RuntimeError("Invalid ``p2p_op_list``. Each op is expected to "
                           "to be of type ``paddle.distributed.P2POp``.")

    backend = _group_map_backend[p2p_op_list[0].group]
    if not all(backend == _group_map_backend[p2p_op.group]
               for p2p_op in p2p_op_list):
        raise RuntimeError("All groups need to use the same backend.")
