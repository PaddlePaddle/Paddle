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

import sys

import paddle

print("Paddle version:", paddle.__version__)
print("Paddle path:", paddle.__file__)
sys.stdout.flush()

edge_index = paddle.arange(17758, dtype='int64')
strided_edge_index = edge_index[::32]
print(strided_edge_index)
print(strided_edge_index[[-1]])
assert strided_edge_index[[-1]][0].item() == 17728
