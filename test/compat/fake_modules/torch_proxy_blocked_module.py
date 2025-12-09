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


def try_import_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def use_torch_specific_fn():
    torch = try_import_torch()

    if torch is None:
        return
    # torch._dynamo.allow_in_graph is a torch specific function, it shouldn't be accessed via proxy
    torch._dynamo.allow_in_graph(lambda x: x)


# Use torch specific function at execute module stage
use_torch_specific_fn()
