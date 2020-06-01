#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define learning rate decay  
from ...fluid.layers import cosine_decay  #DEFINE_ALIAS
from ...fluid.layers import exponential_decay  #DEFINE_ALIAS
from ...fluid.layers import inverse_time_decay  #DEFINE_ALIAS
from ...fluid.layers import natural_exp_decay  #DEFINE_ALIAS
from ...fluid.layers import noam_decay  #DEFINE_ALIAS
from ...fluid.layers import piecewise_decay  #DEFINE_ALIAS
from ...fluid.layers import polynomial_decay  #DEFINE_ALIAS
from ...fluid.layers import linear_lr_warmup  #DEFINE_ALIAS

__all__ = [
    'cosine_decay', 'exponential_decay', 'inverse_time_decay',
    'natural_exp_decay', 'noam_decay', 'piecewise_decay', 'polynomial_decay',
    'linear_lr_warmup'
]
