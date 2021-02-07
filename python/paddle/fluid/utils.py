#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import six
from .framework import Variable


def _contain_var(list_or_tuple):
    """
    Check whether list or tuple contains variable.
    """
    for item in list_or_tuple:
        if isinstance(item, Variable):
            return True
    return False


def _convert_to_tensor_list(old_list, dtype="int32"):
    """
    Converts all elements of a list to Variable.
    """
    from .layers.tensor import fill_constant
    tensor_list = []

    for ele in old_list:
        if isinstance(ele, Variable):
            ele.stop_gradient = True
            tensor_list.append(ele)
        else:
            assert isinstance(ele, six.integer_types)
            ele_tensor = fill_constant([1], dtype, ele, force_cpu=True)
            tensor_list.append(ele_tensor)

    return tensor_list
