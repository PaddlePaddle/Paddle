# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.framework as framework
import paddle.fluid.core as core


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

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            if dist.get_rank() == 0:
                data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            else:
                data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            print(data)
            # [[5, 7, 9], [5, 7, 9]] (2 GPUs)
    """
    SUM = 0
    MAX = 1
    MIN = 2
    PROD = 3
    AVG = 4


def _get_reduce_op(reduce_op, func_name):
    if framework.in_dygraph_mode():
        if reduce_op == ReduceOp.SUM:
            return core.ReduceOp.SUM
        elif reduce_op == ReduceOp.MAX:
            return core.ReduceOp.MAX
        elif reduce_op == ReduceOp.MIN:
            return core.ReduceOp.MIN
        elif reduce_op == ReduceOp.PROD:
            return core.ReduceOp.PRODUCT
    else:
        if reduce_op == ReduceOp.SUM:
            return 'c_allreduce_sum'
        elif reduce_op == ReduceOp.MAX:
            return 'c_allreduce_max'
        elif reduce_op == ReduceOp.MIN:
            return 'c_allreduce_min'
        elif reduce_op == ReduceOp.PROD:
            return 'c_allreduce_prod'

    raise ValueError("Unknown reduce_op type for {}.".format(func_name))
