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
import paddle.fluid.core as core

__all__ = [
    "_get_group_rank",
    "_check_single_tensor",
]


def _get_group_rank(global_rank, group=None):
    return global_rank if group is None else group.get_group_rank(global_rank)


def _check_single_tensor(tensor, tensor_name):
    if not isinstance(tensor, (core.eager.Tensor, paddle.Tensor)):
        raise RuntimeError("Invalid function argument. Expected parameter {}"
                           "to be of type paddle.Tensor, but it's {}".format(
                               tensor_name, type(tensor)))
