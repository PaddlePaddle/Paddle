# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# type mapping: types in yaml -> types in c++ API
input_types_map = {
    'Tensor': 'const Tensor&',
    'Tensor[]': 'const std::vector<Tensor>&',
}

optional_input_types_map = {
    'Tensor': 'const paddle::optional<Tensor>&',
    'Tensor[]': 'const paddle::optional<std::vector<Tensor>>&',
}

attr_types_map = {
    # special types
    'IntArray': 'const IntArray&',
    'Scalar': 'const Scalar&',
    'Scalar(bool)': 'const Scalar&',
    'Scalar(int)': 'const Scalar&',
    'Scalar(int64_t)': 'const Scalar&',
    'Scalar(float)': 'const Scalar&',
    'Scalar[]': 'const std::vector<Scalar>&',
    'Place': 'Place',
    'DataLayout': 'DataLayout',
    'DataType': 'DataType',
    # scalar types
    'bool': 'bool',
    'int': 'int',
    'int64_t': 'int64_t',
    'float': 'float',
    'double': 'double',
    'str': 'const std::string&',
    # vector types
    'bool[]': 'const std::vector<bool>&',
    'int[]': 'const std::vector<int>&',
    'int64_t[]': 'const std::vector<int64_t>&',
    'float[]': 'const std::vector<float>&',
    'double[]': 'const std::vector<double>&',
    'str[]': 'const std::vector<<std::string>&',
}

opmaker_attr_types_map = {
    # special types
    'IntArray': 'std::vector<int64_t>',
    'Scalar': 'float',
    'Scalar(bool)': 'bool',
    'Scalar(int)': 'int',
    'Scalar(int64_t)': 'int64_t',
    'Scalar(float)': 'float',
    'Scalar[]': 'std::vector<Scalar>',
    'Place': 'int',
    'DataLayout': 'int',
    'DataType': 'int',
    # scalar types
    'bool': 'bool',
    'int': 'int',
    'int64_t': 'int64_t',
    'float': 'float',
    'double': 'double',
    'str': 'std::string',
    # vector types
    'bool[]': 'std::vector<bool>',
    'int[]': 'std::vector<int>',
    'int64_t[]': 'std::vector<int64_t>',
    'float[]': 'std::vector<float>',
    'double[]': 'std::vector<double>',
    'str[]': 'std::vector<<std::string>',
}

output_type_map = {'Tensor': 'Tensor', 'Tensor[]': 'std::vector<Tensor>'}

# ------------------------------ phi attr ------------------------------
phi_attr_types_map = attr_types_map.copy()
phi_attr_types_map.update(
    {
        'IntArray': 'const phi::IntArray&',
        'Scalar': 'const phi::Scalar&',
        'Scalar[]': 'std::vector<phi::Scalar>&',
    }
)

# --------------------------- phi dense tensor ---------------------------
# type mapping to phi, used in implementation
dense_input_types_map = {
    'Tensor': 'const phi::DenseTensor&',
    'Tensor[]': 'const std::vector<const phi::DenseTensor*>&',
}

dense_optional_input_types_map = {
    'Tensor': 'paddle::optional<const phi::DenseTensor&>',
    'Tensor[]': 'paddle::optional<const std::vector<phi::DenseTensor>&>',
}

dense_output_types_map = {
    'Tensor': 'phi::DenseTensor*',
    'Tensor[]': 'std::vector<phi::DenseTensor*>',
}

# ---------------------- phi selected rows------------------------------
# type mapping to phi, used in implementation
sr_input_types_map = {
    'Tensor': 'const phi::SelectedRows&',
}

sr_optional_input_types_map = {
    'Tensor': 'const paddle::optional<phi::SelectedRows>&',
}

sr_output_types_map = {
    'Tensor': 'phi::SelectedRows*',
}
