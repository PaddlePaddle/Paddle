# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.trainer.PyDataProvider2 import \
    InputType, dense_vector, sparse_binary_vector,\
    sparse_vector, integer_value, integer_value_sequence

__all__ = [
    'InputType', 'dense_vector', 'sparse_binary_vector', 'sparse_vector',
    'integer_value', 'integer_value_sequence'
]
