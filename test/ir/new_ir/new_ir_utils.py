# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import LayerHelper, core


def new_ir_data(
    name,
    shape,
    dtype=None,
):
    helper = LayerHelper('data', **locals())
    out = helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=paddle.get_default_dtype(),
        type=core.VarDesc.VarType.LOD_TENSOR,
        stop_gradient=True,
        is_data=True,
        need_check_feed=True,
    )
    helper.append_op(
        type='data',
        inputs={},
        outputs={'out': out},
        attrs={
            'index': 0,
            'dtype': 0,
            'place': 0,
            'name': name,
        },
    )
    return out
