#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from ..registry import register_layer
__all__ = [
    'mean', 'mul', 'dropout', 'reshape', 'sigmoid', 'scale', 'transpose',
    'sigmoid_cross_entropy_with_logits', 'elementwise_add', 'elementwise_div',
    'elementwise_sub', 'elementwise_mul', 'clip', 'abs', 'sequence_softmax'
]

for _OP in set(__all__):
    globals()[_OP] = register_layer(_OP)
