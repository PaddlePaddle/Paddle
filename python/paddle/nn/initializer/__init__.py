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

# TODO: define the initializers to create a Parameter in neural network

from ...fluid.initializer import Bilinear  #DEFINE_ALIAS
from ...fluid.initializer import Constant  #DEFINE_ALIAS
from ...fluid.initializer import MSRA  #DEFINE_ALIAS
from ...fluid.initializer import Normal  #DEFINE_ALIAS
from ...fluid.initializer import TruncatedNormal  #DEFINE_ALIAS
from ...fluid.initializer import Uniform  #DEFINE_ALIAS
from ...fluid.initializer import Xavier  #DEFINE_ALIAS

__all__ = [
    'Bilinear',
    'Constant',
    'MSRA',
    'Normal',
    'TruncatedNormal',
    'Uniform',
    'Xavier',
]
