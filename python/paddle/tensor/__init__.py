# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
from ..fluid.layer_helper import LayerHelper

__all__ = ["index_select"]


def index_select(input, index, dim):
    helper = LayerHelper("index_select", **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='index_select',
        inputs={'X': input,
                'index': index},
        outputs={'Out': out},
        attrs={'dim': dim})
    return out
