# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.tensor import magic_method_func

from ..base.dygraph.generated_tensor_methods_patch import _all_method_op_map
from . import Value


def monkey_patch_generated_methods_for_value():
    magic_method_dict = {v: k for k, v in magic_method_func}

    for module_path, method_name, method in _all_method_op_map:
        if module_path != 'paddle.Tensor':
            continue
        setattr(Value, method_name, method)
        magic_name = magic_method_dict.get(method_name)
        if magic_name:
            setattr(Value, magic_name, method)
