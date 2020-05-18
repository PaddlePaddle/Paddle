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

# TODO: define the control flow api  
from ..fluid.layers import case  #DEFINE_ALIAS
from ..fluid.layers import cond  #DEFINE_ALIAS
from ..fluid.layers import while_loop  #DEFINE_ALIAS

from ..fluid.layers import switch_case  #DEFINE_ALIAS

__all__ = [
    'case',
    'cond',
    #       'DynamicRNN',
    #       'StaticRNN',
    'switch_case',
    'while_loop',
    #       'rnn'
]
