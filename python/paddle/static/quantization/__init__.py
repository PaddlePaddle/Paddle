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

from ...quantization.slim.quantization.quantization_pass import (
    QuantizationTransformPass,
)
from ...quantization.slim.quantization.quantization_pass import (
    QuantizationFreezePass,
)
from ...quantization.slim.quantization.quantization_pass import (
    ConvertToInt8Pass,
)
from ...quantization.slim.quantization.quantization_pass import (
    TransformForMobilePass,
)
from ...quantization.slim.quantization.quantization_pass import (
    OutScaleForTrainingPass,
)
from ...quantization.slim.quantization.quantization_pass import (
    OutScaleForInferencePass,
)
from ...quantization.slim.quantization.quantization_pass import (
    AddQuantDequantPass,
)
from ...quantization.slim.quantization.quantization_pass import (
    ReplaceFakeQuantDequantPass,
)
from ...quantization.slim.quantization.quantization_pass import QuantWeightPass
from ...quantization.slim.quantization.quantization_pass import (
    QuantizationTransformPassV2,
)
from ...quantization.slim.quantization.quantization_pass import (
    AddQuantDequantPassV2,
)
from ...quantization.slim.quantization.quant_int8_mkldnn_pass import (
    QuantInt8MkldnnPass,
)
from ...quantization.slim.quantization.quant2_int8_mkldnn_pass import (
    Quant2Int8MkldnnPass,
)

from ...quantization.slim.quantization.post_training_quantization import (
    PostTrainingQuantization,
)
from ...quantization.slim.quantization.post_training_quantization import (
    PostTrainingQuantizationProgram,
)
from ...quantization.slim.quantization.post_training_quantization import (
    WeightQuantization,
)
