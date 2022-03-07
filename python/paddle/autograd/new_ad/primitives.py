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

from typing import Any
from .runner import get_current_runner


class Primitive(object):
    """ Primitive OP.
  
    In instance of `Primitive` identifies a primitive and provides
    interfaces for using the primitive.

    """

    def __init__(self, optype) -> None:
        self.optype = optype

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        runner = get_current_runner()
        runner.run_op(self, *args, **kwargs)


RESHAPE = Primitive('reshape_p')
BCAST = Primitive('broadcast_p')
REDUCE = Primitive('reduce_p')
TRANSPOSE = Primitive('transpose_p')
SPLIT = Primitive('split_p')
CONCAT = Primitive('concat_p')
SLISELECT = Primitive('slice_select_p')
SLIASSIGN = Primitive('slice_assign_p')
INDSELECT = Primitive('index_select_p')
INDASSIGN = Primitive('index_assign_p')
ADD = Primitive('add_p')
SUB = Primitive('sub_p')
MUL = Primitive('mul_p')
DIV = Primitive('div_p')
SQRT = Primitive('sqrt_p')
TANH = Primitive('tanh_p')
MATMUL = Primitive('matmul_p')
FILL = Primitive('fill_constant_p')
