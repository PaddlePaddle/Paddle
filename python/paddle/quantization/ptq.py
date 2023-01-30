# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy

from paddle.nn import Layer

from .config import QuantConfig
from .quantize import Quantization


class PTQ(Quantization):
    """
    Applying post training quantization to the model.
    """

    def __init__(self, config: QuantConfig):
        super(PTQ, self).__init__(config)

    def quantize(self, model: Layer, inplace=False):
        _model = model
        if not inplace:
            _model = copy.deepcopy(model)
            _model.eval()
        assert (
            not model.training
        ), "Post-Training Quantization shoud not work on training models. Please set evaluation mode by model.eval()."
        self._config._specify(_model)
        print(f"config: {self._config.details()}")
        self._convert_to_quant_layers(_model, self._config)
        self._insert_activation_observers(_model, self._config)
        return _model
