# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from ..core.strategy import Strategy

__all__ = ['QuantizationStrategy']


class QuantizationStrategy(Strategy):
    """
    The strategy for Quantization.
    """

    def __init__(self,
                 quantizer,
                 start_epoch=0,
                 end_epoch=10,
                 dirname=None,
                 target_device='mobile',
                 save_as_int8=True):
        super(QuantizationStrategy, self).__init__(start_epoch, end_epoch)
        self.quantizer = quantizer
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.dirname = dirname
        self.target_device = target_device
        self.save_as_int8 = save_as_int8

    def on_compress_begin(self, context):
        super(QuantizationStrategy, self).on_compress_begin(context)
        self.quantizer.quantize(context.train_graph, context.scope)

    def on_compress_end(self, context):
        super(QuantizationStrategy, self).on_compress_end(context)
        self.quantizer.convert_model(context.graph, context.place,
                                     context.scope, context.feeds,
                                     context.fetches, self.dirname, context.exe,
                                     self.target_device, self.save_as_int8)
