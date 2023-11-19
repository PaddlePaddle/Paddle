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

OP_GET_KERNEL_TYPE_FOR_VAR_TEMPLATE = """
std::tuple<phi::DataType, phi::Backend> ParseKernelKey() {{
  VLOG(4) << "Parse kernel key for op: {op_name}";
  return {op_name}ParseKernelKey(operation());
}}
"""


def gen_parse_kernel_key_str(op_class_name):
    return OP_GET_KERNEL_TYPE_FOR_VAR_TEMPLATE.format(op_name=op_class_name)
