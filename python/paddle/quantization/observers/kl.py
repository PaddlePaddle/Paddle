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

import paddle
from ..factory import ObserverFactory
from ..observer import BaseObserver

__all__ = ["KLObserver"]


class KLObserver(ObserverFactory):
    def __init__(self, quant_bits=8):
        super(KLObserver, self).__init__(quant_bits=quant_bits)

    def get_class(self):
        return KLObserverLayer


class KLObserverLayer(BaseObserver):
    """
    Per-tensor abs max quantizer.
    """

    def __init__(self, layer, quant_bits=8):
        super(KLObserverLayer, self).__init__()
        self._quant_bits = quant_bits

    def forward(self, input):
        abs_max_val = float(paddle.max(paddle.abs(input)).numpy())
        self.abs_max_val = max(abs_max_val, self.abs_max_val)

    def cal_thresholds(self):
        self.thresholds = self.abs_max_val

    def bit_length(self):
        return self._quant_bits

    def quant_axis(self):
        return 0

    def scales(self):
        return None

    def zero_points(self):
        return None
