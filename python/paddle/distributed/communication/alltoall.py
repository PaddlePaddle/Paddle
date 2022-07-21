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

__all__ = ["alltoall", "alltoall_single"]


def _dygraph_all2all(in_tensor_list,
                     out_tensor_list,
                     group=None,
                     use_calc_stream=True):
    temp = paddle.concat(in_tensor_list, axis=0)
    nranks = len(in_tensor_list)
    group = _get_default_group() if group is None else group
    if len(out_tensor_list) == 0:
        tensor_shape = list(in_tensor_list[0].shape)
        tensor_shape[0] *= nranks
        out = paddle.empty(tensor_shape, in_tensor_list[0].dtype)
    else:
        out = paddle.concat(out_tensor_list, axis=0)
    task = group.process_group.alltoall(temp, out)
    task.wait()
    out_tensor_list.clear()
    out_tensor_list.extend(paddle.split(out, nranks, 0))


def _static_all2all(in_tensor_list,
                    out_tensor_list,
                    group=None,
                    use_calc_stream=True):
    temp = paddle.concat(in_tensor_list, axis=0)
    nranks = len(in_tensor_list)
    ring_id = 0 if group is None else group.id

    if _non_static_mode():
        out = _C_ops.alltoall(temp, 'use_calc_stream', use_calc_stream,
                              'ring_id', ring_id)
    else:
        op_type = 'alltoall'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(
            dtype=in_tensor_list[0].dtype)

        if not isinstance(in_tensor_list, list):
            raise ValueError("The type of 'in_tensor_list' for all_to_all "
                             "should be list.")
        for elem in in_tensor_list:
            check_variable_and_dtype(
                elem, 'in_tensor_list',
                ['float16', 'float32', 'float64', 'int32', 'int64'],
                'all_to_all')
        if not isinstance(out_tensor_list, list):
            raise ValueError("The type of 'out_tensor_list' for all_to_all "
                             "should be list.")
        if len(out_tensor_list) != 0:
            raise ValueError("The 'out_tensor_list' for all_to_all "
                             "must be an empty list.")
        helper.append_op(type=op_type,
                         inputs={'X': [temp]},
                         outputs={'Out': [out]},
                         attrs={
                             'ring_id': ring_id,
                             'use_calc_stream': use_calc_stream,
                         })

    out_tensor_list.extend(paddle.split(out, nranks, 0))


def alltoall(in_tensor_list, out_tensor_list, group=None, use_calc_stream=True):
    """
    Scatter tensors in in_tensor_list to all participators averagely and gather the result tensors in out_tensor_list.
    As shown below, the in_tensor_list in GPU0 includes 0_0 and 0_1, and GPU1 includes 1_0 and 1_1.
    Through alltoall operator, the 0_0 in GPU0 will be sent to GPU0 and 0_1 to GPU1, 1_0 in GPU1 sent to GPU0 and 1_1 to GPU1.
    Finally the out_tensor_list in GPU0 includes 0_0 and 1_0, and GPU1 includes 0_1 and 1_1.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/alltoall.png
        :width: 800
        :alt: alltoall
        :align: center

    Args:
        in_tensor_list (list): A list of input Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        out_tensor_list (list): A list of output Tensors. The data type of its elements should be the same as the
            data type of the input Tensors.
        group (Group, optional): The group instance return by new_group or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculation stream (True) or communication stream. Default: True.
    
    Returns:
        None.
    
    Examples:
        .. code-block:: python

            # required: distributed
            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env
            
            init_parallel_env()
            out_tensor_list = []
            if paddle.distributed.ParallelEnv().rank == 0:
                np_data1 = np.array([[1, 2, 3], [4, 5, 6]])
                np_data2 = np.array([[7, 8, 9], [10, 11, 12]])
            else:
                np_data1 = np.array([[13, 14, 15], [16, 17, 18]])
                np_data2 = np.array([[19, 20, 21], [22, 23, 24]])
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.alltoall([data1, data2], out_tensor_list)
            # out for rank 0: [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]]
            # out for rank 1: [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]]
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        _dygraph_all2all(in_tensor_list, out_tensor_list, group,
                         use_calc_stream)
        return

    _static_all2all(in_tensor_list, out_tensor_list, group, use_calc_stream)


def alltoall_single(in_tensor,
                    out_tensor,
                    in_split_sizes=None,
                    out_split_sizes=None,
                    group=None,
                    use_calc_stream=True):
    """
    Scatter a single input tensor to all participators and gather the received tensors in out_tensor.

    .. note::
        ``alltoall_single`` is only supported in eager mode.

    Args:
        in_tensor (Tensor): Input tensor. The data type should be float16, float32, float64, int32 or int64.
        out_tensor (Tensor): Output Tensor. The data type should be the same as the data type of the input Tensor.
        in_split_sizes (list[int], optional): Split sizes of ``in_tensor`` for dim[0]. If not given, dim[0] of ``in_tensor`` 
            must be divisible by group size and ``in_tensor`` will be scattered averagely to all participators. Default: None.
        out_split_sizes (list[int], optional): Split sizes of ``out_tensor`` for dim[0]. If not given, dim[0] of ``out_tensor`` 
            must be divisible by group size and ``out_tensor`` will be gathered averagely from all participators. Default: None.
        group (Group, optional): The group instance return by ``new_group`` or None for global default group. Default: None.
        use_calc_stream (bool, optional): Whether to use calculation stream (True) or communication stream. Default: True.
    
    Returns:
        None, if ``use_calc_stream`` is set to ``True``; ``Task`` of ``group``, if ``use_calc_stream`` is set to ``False``.
    
    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            rank = dist.get_rank()
            size = dist.get_world_size()

            # case 1
            input = paddle.arange(2, dtype='int64') + rank * 2
            # input for rank 0: [0, 1]
            # input for rank 1: [2, 3]
            
            output = paddle.empty([2], dtype='int64')
            dist.alltoall_single(input, output)
            # output for rank 0: [0, 2]
            # output for rank 1: [1, 3]

            # case 2
            in_split_sizes = [i + 1 for i in range(size)]
            # in_split_sizes for rank 0: [1, 2] and for rank 1: [1, 2]
            out_split_sizes = [rank + 1 for i in range(size)]
            # out_split_sizes for rank 0: [1, 1] and for rank 1: [2, 2]

            input = paddle.ones([sum(in_split_sizes), size], dtype='float32') * rank
            # input for rank 0: [[0., 0.], [0., 0.], [0., 0.]]
            # input for rank 1: [[1., 1.], [1., 1.], [1., 1.]]
            output = paddle.empty([(rank + 1) * size, size], dtype='float32')

            group = dist.new_group([0, 1])
            task = dist.alltoall_single(input,
                                        output,
                                        in_split_sizes,
                                        out_split_sizes,
                                        use_calc_stream=False,
                                        group=group)
            task.wait()
            # output for rank 0: [[0., 0.], [1., 1.]]
            # output for rank 1: [[0., 0.], [0., 0.], [1., 1.], [1., 1.]]

    """
    if group is not None and not group.is_member():
        return

    assert in_dygraph_mode(), "Only suppport alltoall_single in eager mode."
    # _check_single_tensor

    group = _get_default_group() if group is None else group
    in_split_sizes = [] if in_split_sizes is None else in_split_sizes
    out_split_sizes = [] if out_split_sizes is None else out_split_sizes

    task = group.process_group.alltoall_single(in_tensor, out_tensor,
                                               in_split_sizes, out_split_sizes)
    if use_calc_stream:
        task.wait()
        return
    else:
        return task
