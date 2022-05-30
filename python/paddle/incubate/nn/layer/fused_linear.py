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

from paddle.nn import Layer
from paddle.incubate.nn import functional as F


class FusedLinear(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 transpose_weight=False,
                 name=None):
        super(FusedLinear, self).__init__()
        if transpose_weight:
            weight_shape = [out_features, in_features]
        else:
            weight_shape = [in_features, out_features]
        dtype = self._helper.get_default_dtype()
        self.weight = self.create_parameter(
            shape=weight_shape, attr=weight_attr, dtype=dtype, is_bias=False)
        self.bias = self.create_parameter(
            shape=[out_features], attr=bias_attr, dtype=dtype, is_bias=True)
        self.transpose_weight = transpose_weight
        self.name = name

    def forward(self, input):
        return F.fused_linear(input, self.weight, self.bias,
                              self.transpose_weight, self.name)
