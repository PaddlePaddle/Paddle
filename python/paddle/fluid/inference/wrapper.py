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

from ..core import AnalysisConfig, PaddleDType, PaddlePlace
from ..core import PaddleInferPredictor, PaddleInferTensor

import numpy as np

DataType = PaddleDType
PlaceType = PaddlePlace
PrecisionType = AnalysisConfig.Precision
Config = AnalysisConfig
Tensor = PaddleInferTensor
Predictor = PaddleInferPredictor


def tensor_copy_from_cpu(self, data):
    '''
    Support input type check based on tensor.copy_from_cpu.
    '''
    if not isinstance(data, np.ndarray):
        raise TypeError(
            "In copy_from_cpu, we only support numpy ndarray data type.")
    self.copy_from_cpu_bind(data)


Tensor.copy_from_cpu = tensor_copy_from_cpu
