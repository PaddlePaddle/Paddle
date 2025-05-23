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

function_to_axpr_atomic = __builtin__function_to_axpr_atomic  # noqa: F821
axpr_atomic_to_function = __builtin__axpr_atomic_to_function  # noqa: F821
axpr_json_str_to_axpr = __builtin__axpr_json_str_to_axpr  # noqa: F821


def axpr_lambda_json_str_to_function(axpr_lambda_json_str):
    axpr_obj = axpr_json_str_to_axpr(axpr_lambda_json_str)
    axpr_atomic = axpr_obj.match(axpr_atomic=lambda x: x)
    return axpr_atomic_to_function(axpr_atomic)


axpr_symbol = __builtin__axpr_symbol  # noqa: F821
axpr_none = __builtin__axpr_none  # noqa: F821
axpr_bool = __builtin__axpr_bool  # noqa: F821
axpr_int = __builtin__axpr_int  # noqa: F821
axpr_float = __builtin__axpr_float  # noqa: F821
axpr_str = __builtin__axpr_str  # noqa: F821
axpr_lambda = __builtin__axpr_lambda  # noqa: F821
axpr_atomic = __builtin__axpr_atomic  # noqa: F821
axpr_call = __builtin__axpr_call  # noqa: F821
