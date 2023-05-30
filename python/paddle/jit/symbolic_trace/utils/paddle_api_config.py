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

import json
import os
import sys
import warnings

paddle_api_file_path = os.path.join(
    os.path.dirname(__file__), "paddle_api_info", "paddle_api.json"
)
with open(paddle_api_file_path, "r") as file:
    paddle_api = json.load(file)

# tensor_methods skipped __iadd__ __isub__, because variable do not support inplace operators
paddle_tensor_method_file_path = os.path.join(
    os.path.dirname(__file__), "paddle_api_info", "paddle_tensor_method.json"
)
# TODO(Aurelius84): Can we automitically parse the apis list from dir(paddle.tensor).
with open(paddle_tensor_method_file_path, "r") as file:
    paddle_tensor_method = json.load(file)

paddle_api_list = set()
for module_name in paddle_api.keys():
    # it should already be imported
    if module_name in sys.modules.keys():
        module = sys.modules[module_name]
        apis = paddle_api[module_name]
        for api in apis:
            if api in module.__dict__.keys():
                obj = module.__dict__[api]
                paddle_api_list.add(obj)
    else:
        warnings.warn(f"{module_name} not imported.")

# TODO(Aurelius84): It seems that we use it to judge 'in_paddle_module()'.
# Bug what does 'is_paddle_module' really means? Is all paddle.xx sub module
# considered as paddle module？
paddle_api_module_prefix = {
    "paddle.nn.functional",
    "paddle.nn.layer.activation",
}

fallback_list = {
    print,
    # paddle.utils.map_structure,
}
