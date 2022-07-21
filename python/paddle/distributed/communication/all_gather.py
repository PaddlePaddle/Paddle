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

__all__ = ["all_gather"]


def _dygraph_all_gather(tensor_list, tensor, group=None, use_calc_stream=True):
    group = _get_default_group() if group is None else group
    if len(tensor_list) == 0:
        tensor_shape = list(tensor.shape)
        tensor_shape[0] *= group.nranks
        out = paddle.empty(tensor_shape, tensor.dtype)
    else:
        out = paddle.concat(tensor_list, axis=0)
    task = group.process_group.all_gather(tensor, out)
    task.wait()
    tensor_list.clear()
    tensor_list.extend(paddle.split(out, group.nranks, 0))


def _static_all_gather(tensor_list, tensor, group=None, use_calc_stream=True):
    ring_id = 0 if group is None else group.id
    nranks = _get_global_group().nranks if group is None else group.nranks

    if _non_static_mode():
        out = _C_ops.c_allgather(tensor, 'use_calc_stream', use_calc_stream,
                                 'ring_id', ring_id, 'nranks', nranks)
    else:
        op_type = 'c_allgather'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
        if not isinstance(tensor_list, list):
            raise ValueError("The type of 'tensor_list' for all_gather "
                             "should be list.")
        for elem in tensor_list:
            check_variable_and_dtype(
                elem, 'tensor_list',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                'all_gather')
        check_variable_and_dtype(
            tensor, 'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64'], 'all_gather')

        helper.append_op(type=op_type,
                         inputs={'X': [tensor]},
                         outputs={'Out': [out]},
                         attrs={
                             'ring_id': ring_id,
                             'use_calc_stream': use_calc_stream,
                             'nranks': nranks
                         })

    tensor_list.extend(paddle.split(out, nranks, 0))


def all_gather(tensor_list, tensor, group=None, use_calc_stream=True):
    """

    Gather tensors from all participators and all get the result. As shown
    below, 4 GPUs each start 4 processes and the data on each GPU is represnted
    by the GPU number. Through the all_gather operator, each GPU will have data
    from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/allgather.png
        :width: 800
        :alt: all_gather
        :align: center

    Args:
        tensor_list (list): A list of output Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
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
            tensor_list = []
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
                np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
                data1 = paddle.to_tensor(np_data1)
                data2 = paddle.to_tensor(np_data2)
                paddle.distributed.all_gather(tensor_list, data1)
            else:
                np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
                np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
                data1 = paddle.to_tensor(np_data1)
                data2 = paddle.to_tensor(np_data2)
                paddle.distributed.all_gather(tensor_list, data2)
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        _dygraph_all_gather(tensor_list, tensor, group, use_calc_stream)
        return

    _static_all_gather(tensor_list, tensor, group, use_calc_stream)
