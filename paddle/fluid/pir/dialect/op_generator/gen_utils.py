# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


def to_pascal_case(s):
    words = s.split("_")
    if s[-1] == "_":
        return "".join([word.capitalize() for word in words]) + "_"
    else:
        return "".join([word.capitalize() for word in words]) + ""


attr_types_map = {
    'IntArray': ['paddle::dialect::IntArrayAttribute', 'IntArray'],
    'Scalar': ['paddle::dialect::ScalarAttribute', 'Scalar'],
    'Scalar(int)': ['paddle::dialect::ScalarAttribute', 'Scalar'],
    'Scalar(int64_t)': ['paddle::dialect::ScalarAttribute', 'Scalar'],
    'Scalar(float)': ['paddle::dialect::ScalarAttribute', 'Scalar'],
    'Scalar(double)': ['paddle::dialect::ScalarAttribute', 'Scalar'],
    'Scalar[]': [
        'pir::ArrayAttribute<paddle::dialect::ScalarAttribute>',
        'const std::vector<Scalar>&',
    ],
    'int': ['pir::Int32Attribute', 'int'],
    'int32_t': ['pir::Int32Attribute', 'int32_t'],
    'int64_t': ['pir::Int64Attribute', 'int64_t'],
    'long': ['pir::LongAttribute', 'long'],
    'size_t': ['pir::Size_tAttribute', 'size_t'],
    'float': ['pir::FloatAttribute', 'float'],
    'float[]': [
        'pir::ArrayAttribute<pir::FloatAttribute>',
        'const std::vector<float>&',
    ],
    'double': ['pir::DoubleAttribute', 'double'],
    'bool': ['pir::BoolAttribute', 'bool'],
    'bool[]': [
        'pir::ArrayAttribute<pir::BoolAttribute>',
        'const std::vector<bool>&',
    ],
    'str': ['pir::StrAttribute', 'const std::string&'],
    'str[]': [
        'pir::ArrayAttribute<pir::StrAttribute>',
        'const std::vector<std::string>&',
    ],
    'Place': ['paddle::dialect::PlaceAttribute', 'const phi::Place&'],
    'DataLayout': [
        'paddle::dialect::DataLayoutAttribute',
        'DataLayout',
    ],
    'DataType': ['paddle::dialect::DataTypeAttribute', 'DataType'],
    'int64_t[]': [
        'pir::ArrayAttribute<pir::Int64Attribute>',
        'const std::vector<int64_t>&',
    ],
    'int[]': [
        'pir::ArrayAttribute<pir::Int32Attribute>',
        'const std::vector<int>&',
    ],
}
