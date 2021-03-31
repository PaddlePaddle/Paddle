#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import os
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import Variable, OpProtoHolder, in_dygraph_mode, convert_np_dtype_to_dtype_
from ..fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ..fluid.layers.tensor import fill_constant
from ..fluid.layers import utils
from ..fluid.dygraph.parallel import prepare_context
import paddle
from .fleet import fleet
import paddle.fluid as fluid
import paddle.fluid.core as core

__all__ = [
    'broadcast',
    'all_reduce',
    'reduce',
    'all_gather',
    'scatter',
    'barrier',
    'split',
    'ReduceOp',
]


class ReduceOp:
    """
    Specify the type of operation used for element-wise reductions.
    It should be one of the following values:

        ReduceOp.SUM

        ReduceOp.MAX

        ReduceOp.MIN

        ReduceOp.PROD

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.distributed import ReduceOp
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.all_reduce(data, op=ReduceOp.SUM)
            out = data.numpy()
            # [[5, 7, 9], [5, 7, 9]]
    """
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3


class _Group():
    """The abstract representation of group."""

    def __init__(self, rank, rank_num):
        self.rank = rank
        self.nranks = rank_num


# NOTE(chenweihang): Lazily initialized global group information
# If we initialize _default_group when import module, it will 
# not update when we use spawn to run multi-process training 
_default_group = None


def _get_global_default_group():
    global _default_group
    if _default_group is None:
        _default_group = _Group(
            int(os.getenv("PADDLE_TRAINER_ID", "0")),
            int(os.getenv("PADDLE_TRAINERS_NUM", "1")))
    return _default_group


def broadcast(tensor, src, group=0):
    """

    Broadcast a tensor from the source to all others.

    Args:
        tensor (Tensor): The Tensor to send if current rank is the source, or the tensor to receive otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank.
        group (int): The process group to work on. It is Optional.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.broadcast(data, 1)
            out = data.numpy()
            # [[1, 2, 3], [1, 2, 3]]
    """
    if in_dygraph_mode():
        return core.ops.c_broadcast(tensor, tensor, 'root', src,
                                    'use_calc_stream', True, 'ring_id', group)

    op_type = 'c_broadcast'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'broadcast')
    if not isinstance(src, int) or not isinstance(group, int):
        raise ValueError("Both the type of 'src' and 'group' for broadcast "
                         "should be int.")

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'root': src,
            'use_calc_stream': True,
            'ring_id': group,
        })


def all_reduce(tensor, op=ReduceOp.SUM, group=0):
    """

    Reduce a tensor over all ranks so that all get the result.

    Args:
        tensor (Tensor): The input Tensor. It also works as the output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used.
        group (int): Optional. The process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.distributed import ReduceOp
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.all_reduce(data)
            out = data.numpy()
            # [[5, 7, 9], [5, 7, 9]]
    """
    if in_dygraph_mode():
        if op == ReduceOp.SUM:
            return core.ops.c_allreduce_sum(tensor, tensor, 'use_calc_stream',
                                            True, 'ring_id', group)
        elif op == ReduceOp.MAX:
            return core.ops.c_allreduce_max(tensor, tensor, 'use_calc_stream',
                                            True, 'ring_id', group)
        elif op == ReduceOp.MIN:
            return core.ops.c_allreduce_min(tensor, tensor, 'use_calc_stream',
                                            True, 'ring_id', group)
        elif op == ReduceOp.PROD:
            return core.ops.c_allreduce_prod(tensor, tensor, 'use_calc_stream',
                                             True, 'ring_id', group)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))

    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not op in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PROD]:
        raise ValueError("The op for all_reduce must be one of educeOp.PROD, "
                         "ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN.")
    if op == ReduceOp.SUM:
        op_type = 'c_allreduce_sum'
    elif op == ReduceOp.MAX:
        op_type = 'c_allreduce_max'
    elif op == ReduceOp.MIN:
        op_type = 'c_allreduce_min'
    elif op == ReduceOp.PROD:
        op_type = 'c_allreduce_prod'
    if not isinstance(group, int):
        raise ValueError("The type of 'group' for all_reduce should be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={'ring_id': group,
               'use_calc_stream': True})


def reduce(tensor, dst, op=ReduceOp.SUM, group=0):
    """

    Reduce a tensor to the destination from all others.

    Args:
        tensor (Tensor): The output Tensor for the destination and the input Tensor otherwise. Its data type
            should be float16, float32, float64, int32 or int64.
        dst (int): The destination rank id.
        op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD): Optional. The operation used.
        group (int): The id of the process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            if paddle.distributed.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.reduce(data, 0)
            out = data.numpy()
            # [[5, 7, 9], [5, 7, 9]]
    """
    if in_dygraph_mode():
        if op == ReduceOp.SUM:
            return core.ops.c_reduce_sum(tensor, tensor, 'use_calc_stream',
                                         True, 'ring_id', group, 'root_id', dst)
        elif op == ReduceOp.MAX:
            return core.ops.c_reduce_max(tensor, tensor, 'use_calc_stream',
                                         True, 'ring_id', group, 'root_id', dst)
        elif op == ReduceOp.MIN:
            return core.ops.c_reduce_min(tensor, tensor, 'use_calc_stream',
                                         True, 'ring_id', group, 'root_id', dst)
        elif op == ReduceOp.PROD:
            return core.ops.c_reduce_prod(tensor, tensor, 'use_calc_stream',
                                          True, 'ring_id', group, 'root_id',
                                          dst)
        else:
            raise ValueError("Unknown parameter: {}.".format(op))

    op_type = 'c_reduce'
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'all_reduce')
    if not op in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PROD]:
        raise ValueError("The op for reduce must be one of educeOp.PROD, "
                         "ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN.")

    if op == ReduceOp.SUM:
        op_type = 'c_reduce_sum'
    elif op == ReduceOp.MAX:
        op_type = 'c_reduce_max'
    elif op == ReduceOp.MIN:
        op_type = 'c_reduce_min'
    elif op == ReduceOp.PROD:
        op_type = 'c_reduce_prod'

    if not isinstance(dst, int) or not isinstance(group, int):
        raise ValueError("Both the type of 'dst' and 'group' for reduce "
                         "should be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': group,
            'use_calc_stream': True,
            'root_id': dst,
        })


def all_gather(tensor_list, tensor, group=0):
    """

    Gather tensors from all participators and all get the result.

    Args:
        tensor_list (list): A list of output Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

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
    op_type = 'c_allgather'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
    _default_group = _get_global_default_group()
    if in_dygraph_mode():
        core.ops.c_allgather(tensor, out, 'use_calc_stream', True, 'ring_id',
                             group, 'nranks', _default_group.nranks)
    else:
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
        if not isinstance(group, int):
            raise ValueError("The type of 'group' for all_gather "
                             "should be int.")
        helper.append_op(
            type=op_type,
            inputs={'X': [tensor]},
            outputs={'Out': [out]},
            attrs={
                'ring_id': group,
                'use_calc_stream': True,
                'nranks': _default_group.nranks
            })

    tensor_list.extend(paddle.split(out, _default_group.nranks, 0))


def scatter(tensor, tensor_list=None, src=0, group=0):
    """

    Scatter a tensor to all participators.

    Args:
        tensor (Tensor): The output Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        tensor_list (list): A list of Tensors to scatter. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32 or int64.
        src (int): The source rank id.
        group (int): The id of the process group to work on.

    Returns:
        None.

    Examples:
        .. code-block:: python

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
    op_type = 'c_scatter'
    _default_group = _get_global_default_group()
    rank = _default_group.rank
    nranks = _default_group.nranks
    if rank != src:
        tensor_list = []
        for _ in range(nranks):
            tensor_list.append(tensor)
    temp = paddle.concat(tensor_list, axis=0)
    if in_dygraph_mode():
        return core.ops.c_scatter(temp, tensor, 'use_calc_stream', True,
                                  'ring_id', group, 'nranks',
                                  _default_group.nranks, 'root', src)
    check_variable_and_dtype(
        tensor, 'tensor', ['float16', 'float32', 'float64', 'int32', 'int64'],
        'scatter')
    if not isinstance(group, int) or not isinstance(src, int):
        raise ValueError("Both the type of 'src' and 'group' for scatter "
                         "should be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [temp]},
        outputs={'Out': [tensor]},
        attrs={
            'ring_id': group,
            'root': src,
            'use_calc_stream': True,
            'nranks': nranks,
        })


def barrier(group=0):
    """

    Barrier among all participators in the group.

    Args:
        group (int): The id of the process group to work on.

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
    op_type = 'barrier'
    temp = fill_constant([1], dtype="int32", value="1")
    if in_dygraph_mode():
        return core.ops.barrier(temp, temp, 'ring_id', group)
    if not isinstance(group, int):
        raise ValueError("The type of 'group' for barrier must be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [temp]},
        outputs={'Out': [temp]},
        attrs={'ring_id': group})


def _parallel_linear(x, num_rows, num_cols, axis, param_attr, bias_attr,
                     gather_out, inner_rank, name):
    """
    Parallel Linear
    """
    if not name:
        name = "fc_by_row_rank_%d" % inner_rank if axis == 0 else "fc_by_col_rank_%d" % inner_rank
    else:
        name = name + "_by_row_rank_%d" % inner_rank if axis == 0 else name + "_by_col_rank_%d" % inner_rank
    linear = paddle.nn.Linear(
        num_rows,
        num_cols,
        weight_attr=param_attr,
        bias_attr=bias_attr,
        name=name)

    weight = linear.weight
    weight.is_distributed = True
    linear_out = linear(x)
    startup_block = paddle.static.default_startup_program().global_block()
    main_block = paddle.static.default_main_program().global_block()
    startup_block.vars[weight.name].is_distributed = True
    main_block.vars[weight.name].is_distributed = True

    if gather_out:
        if axis == 0:
            paddle.distributed.all_reduce(linear_out, group=0)
        else:
            output = []
            paddle.distributed.all_gather(output, linear_out, group=0)
            linear_out = paddle.concat(output, axis=len(linear_out.shape) - 1)
    return linear_out


def _parallel_embedding(x, per_part_embeddings, origin_size, param_attr,
                        inner_rank, num_partitions, name):
    """
    Parallel Embedding
    """
    if not name:
        name = "emb_rank_%d" % inner_rank
    else:
        name = name + "_rank_%d" % inner_rank

    origin_num_embeddings = origin_size[0]
    embedding = paddle.nn.Embedding(
        per_part_embeddings,
        origin_size[1],
        padding_idx=per_part_embeddings - 1,
        sparse=False,
        weight_attr=param_attr,
        name=name)

    origin_input_shape = x.shape
    if len(origin_input_shape) == 2:
        x = paddle.unsqueeze(x, axis=-1)
    else:
        assert origin_input_shape[-1] == 1, (
            "The last dimension size of x must be 1.")
    x_shard = paddle.shard_index(x, origin_num_embeddings, num_partitions,
                                 inner_rank, per_part_embeddings - 1)
    if len(origin_input_shape) == 2:
        x_shard = paddle.squeeze(x_shard, axis=-1)

    embedding.weight.is_distributed = True
    emb_out = embedding(x_shard)
    startup_block = paddle.static.default_startup_program().global_block()
    main_block = paddle.static.default_main_program().global_block()
    startup_block.vars[embedding.weight.name].is_distributed = True
    main_block.vars[embedding.weight.name].is_distributed = True
    paddle.distributed.all_reduce(emb_out, group=0)
    return emb_out


def split(x,
          size,
          operation,
          axis=0,
          num_partitions=1,
          gather_out=True,
          weight_attr=None,
          bias_attr=None,
          name=None):
    """

    Split the weight of the specified operation into multiple devices
    and do the computation in parallel.

    Now the following three cases are supported.

    Case 1: Parallel Embedding
        The weight of the embedding operation is a NxM matrix with N rows and M columns.
        With parallel embedding, the weight is split into num_partitions partitions, each
        of which is a matrix with (N/num_partitions + 1) rows and M column where the last
        row as the padding idx.
        
        Suppose we split the NxM weight into two partitons on device_0 and device_1
        respectively. Then, one each device, the final weight has (N/2 + 1) rows with the
        index range from 0 to N/2. On device_0, all values in the input within [0, N/2 -1]
        keep unchanged and all other values are changed to N/2 which is the padding index and
        are mapped to all zeros after embedding. In the same way, on device_1, the value V in the
        input within [N/2, N-1] will be changed to (V - N/2), and all other values are changed
        to N/2 and are mapped to all zeros after embedding. Finally, the results on the two
        devices are sum-reduced.

    Case 2: Row Parallel Linear
        The weight of the linear operation is a NxM matrix with N rows and M columns.
        With row parallel linear, the weight is split into num_partitions partitions, each
        of which is a matrix with N/num_partitions rows and M column.

    Case 3: Column Parallel Linear
        The weight of the linear operation is a NxM matrix with N rows and M columns.
        With column parallel linear, the weight is split into num_paratitions partitions, each
        of which is a matrix with N rows and M/num_partitions column.

    Args:
        x (Tensor): Input tensor. It's data type should be float16, float32, float64, int32 or int64.
        size (list|tuple): A list or tuple with two elements indicating the shape of the weight.
        operation (str): The name of the operation. The supported operations are 'linear' and 'embedding'.
        axis (int, Optional): Indicate along which axis to split the weight. Default: 0.
        num_partitions (int, Optional): How many parts the weight is partitioned. Default: 1.
        gather_out (bool, Optional): Whether to gather the output after computation. By default, the output
            on each partitions will be gathered after computation. Default: True.
        weight_attr (ParamAttr, Optional): The parameter attribute for the learnable
            weights(Parameter) of the specified operation. Default: None.
        bias_attr (ParamAttr, Optional): The parameter attribute for the bias
            of the specified operation. Default: None.
        name (str, Optional): The default value is None. Normally there is no need for user to set this
            property. Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            data = paddle.randint(0, 8, shape=[10,4])
            emb_out = padle.distributed.split(
                data,
                (8, 8),
                operation="embedding",
                num_partitions=2)
    """
    assert isinstance(size, (list, tuple)), (
        "The type of size for "
        "paddle.distributed.split must be list or tuple.")
    assert len(size) == 2, ("Number of elements in size of "
                            "paddle.distributed.split must be two.")
    assert isinstance(operation, str), ("The type of operation for "
                                        "paddle.distributed.split must be str.")
    supported_operations = [
        'linear',
        'embedding',
    ]
    assert operation in supported_operations, (
        "The operation for "
        "paddle.distributed.split must be one of {}.".format(
            supported_operations))
    if in_dygraph_mode():
        rank = paddle.distributed.get_rank()
        nranks = paddle.distributed.get_world_size()
    else:
        assert fleet._role_maker, ("To use paddle.distributed.split, "
                                   "you must call fleet.init() firstly.")
        rank = fleet.worker_index()
        nranks = fleet.worker_num()

    # rank within a model parallel group
    inner_rank = rank % num_partitions

    if operation == "embedding":
        assert axis == 0, ("We only support to split the weight of embedding "
                           "along the first axis now.")
        per_part_size = (size[0] + num_partitions - 1) // num_partitions
        last_part_size = size[0] - per_part_size * (num_partitions - 1)
        if inner_rank == num_partitions - 1: per_part_size = last_part_size
        per_part_size += 1  # make the last row as the padding index

        emb_out = _parallel_embedding(x, per_part_size, size, weight_attr,
                                      inner_rank, num_partitions, name)
        return emb_out
    else:
        if axis == 0:
            assert size[0] % num_partitions == 0, (
                "Number of rows of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[0],
                                                           num_partitions))
            per_part_size = size[0] // num_partitions
            linear_size = (per_part_size, size[1])
            assert x.shape[-1] == per_part_size, (
                "The width ({}) of the input "
                "x must be equal to the height ({}) of the weight. Maybe you "
                "should split the input x using paddle.split.".format(
                    x.shape[-1], per_part_size))

        elif axis == 1:
            assert size[1] % num_partitions == 0, (
                "Number of column of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[1],
                                                           num_partitions))
            per_part_size = size[1] // num_partitions
            linear_size = (size[0], per_part_size)
        else:
            raise ValueError("The value of axis must be 0 or 1, but the value "
                             "given is {}.".format(axis))

        linear_out = _parallel_linear(
            x,
            linear_size[0],
            linear_size[1],
            axis,
            weight_attr,
            bias_attr,
            gather_out,
            inner_rank,
            name=name)
        return linear_out
