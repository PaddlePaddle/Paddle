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


class KernelArgTranslator:
    def __init__(self, param_struct_name):
        self.param_struct_name = param_struct_name

    def get_kernel_arg_name(self, var_name):
        return var_name

    def get_param_struct_field_name(self, var_name):
        return var_name

    def get_param_struct_init_name(self, var_name):
        return f"{self.param_struct_name}.{var_name}"

    def get_use_name(self, var_name):
        return f"{self.param_struct_name}.{var_name}"
