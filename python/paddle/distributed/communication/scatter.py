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
from .group import _get_default_group, _get_global_group

__all__ = ["scatter"]

def _dygraph_scatter(inp, out, group, gsrc, use_calc_stream):
    task = group.process_group.scatter(inp, out, gsrc)
    if use_calc_stream:
        task.wait()
        return None
    else:
        return task

def _static_scatter(inp, out, ring_id, gsrc, nranks, use_calc_stream):
    if _non_static_mode():
        return _C_ops.c_scatter(inp, out, 'use_calc_stream',
                                use_calc_stream, 'ring_id', ring_id, 'nranks',
                                nranks, 'root', gsrc)
    op_type = 'c_scatter'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'scatter')
    helper = LayerHelper(op_type, **locals())
    helper.append_op(type=op_type,
                     inputs={'X': [inp]},
                     outputs={'Out': [out]},
                     attrs={
                         'ring_id': ring_id,
                         'root': gsrc,
                         'use_calc_stream': use_calc_stream,
                         'nranks': nranks,
                     })
    

def scatter(tensor, tensor_list=None, src=0, group=None, use_calc_stream=True):
    """

    Scatter a tensor to all participators. As shown below, 4 GPUs each start 4 processes and the source of the scatter
    is GPU0. Through scatter operator, the data in GPU0 will be sent to all GPUs averagely.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/scatter.png
        :width: 800
        :alt: scatter
        :align: center

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        tensor_list (list|tuple): A list/tuple of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64. Default value is None.
        src (int): The source rank id. Default value is 0.
        group (Group): The group instance return by new_group or None for global default group.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data1 = np.array([7, 8, 9])
                np_data2 = np.array([10, 11, 12])
            else:
                np_data1 = np.array([1, 2, 3])
                np_data2 = np.array([4, 5, 6])
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            if paddle.distributed.ParallelEnv().local_rank == 0:
                paddle.distributed.scatter(data1, src=1)
            else:
                paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
            out = data1.numpy()
    """
    if group is not None and not group.is_member():
        return

    if not isinstance(src, int):
        raise ValueError("src should be int.")

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        gsrc = group.get_group_rank(src)
        rank = group.rank
        nranks = group.nranks
    else:
        ring_id = 0 if group is None else group.id
        gsrc = src if group is None else group.get_group_rank(src)
        rank = _get_global_group().rank if group is None else group.rank
        nranks = _get_global_group().nranks if group is None else group.nranks
    assert gsrc >= 0, ("src rank out of group, need global rank")

    if rank != gsrc:
        tensor_list = []
        for _ in range(nranks):
            tensor_list.append(tensor)
    temp = paddle.concat(tensor_list, axis=0)
    if in_dygraph_mode():
        return  _dygraph_scatter(temp, tensor, group, gsrc, use_calc_stream)

    _static_scatter(temp, tensor, ring_id, gsrc, nranks, use_calc_stream)
