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


class CodeGenValue:
    def __init__(self, pir_type, var_name):
        self.pir_type = pir_type
        self.var_name = var_name
        self.const_value = None

    def get_dtype(self):
        def convert_to_dtype(pir_dtype, shape, data_layout):
            return pir_dtype.convert_to_dtype()

        return self.pir_type.match(t_dtensor=convert_to_dtype)

    def is_dense_tensor_type(self):
        return self.pir_type.get_type_name() == "t_dtensor"
