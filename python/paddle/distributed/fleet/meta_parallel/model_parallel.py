#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.dygraph.layers import Layer
from .meta_parallel_base import MetaParallelBase
from ..utils.hybrid_parallel_util import *


class ModelParallel(MetaParallelBase):
    def __init__(self, layers, hcg, **kwargs):
        super(ModelParallel, self).__init__(layers, hcg, **kwargs)

    def _prepare_for_model(self):
        broadcast_mp_parameters(self._layers, self._hcg)
        broadcast_dp_parameters(self._layers, self._hcg)

    def _pre_forward(self, *inputs, **kwargs):
        return broadcast_input_data(self._hcg, *inputs, **kwargs)
