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

# TODO: define functions of linear algebra   
# __all__ = ['matmul', 
#            'dot',
#            'einsum',
#            'morm',
#            'transpose',
#            'dist',
#            't',
#            'cross',
#            'cholesky',
#            'tensordot']
from __future__ import print_function

from ..fluid.layer_helper import LayerHelper

__all__ = ['cross']


def cross(input, other, dim=-1):
    helper = LayerHelper("cross", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='cross',
        inputs={'X': input,
                'Y': other},
        outputs={'Out': out},
        attrs={'dim': dim})
    return out
