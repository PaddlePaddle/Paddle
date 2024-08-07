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

from .post_training_quantization import (  # noqa: F401
    PostTrainingQuantization,
    PostTrainingQuantizationProgram,
    WeightQuantization,
)
from .quant2_int8_onednn_pass import (  # noqa: F401
    Quant2Int8MkldnnPass,
)
from .quant_int8_onednn_pass import (  # noqa: F401
    QuantInt8MkldnnPass,
)
from .quanter import (  # noqa: F401
    convert,
    quant_aware,
)
from .quantization_pass import (  # noqa: F401
    AddQuantDequantForInferencePass,
    AddQuantDequantPass,
    AddQuantDequantPassV2,
    ConvertToInt8Pass,
    OutScaleForInferencePass,
    OutScaleForTrainingPass,
    QuantizationFreezePass,
    QuantizationTransformPass,
    QuantizationTransformPassV2,
    QuantWeightPass,
    ReplaceFakeQuantDequantPass,
    TransformForMobilePass,
)
