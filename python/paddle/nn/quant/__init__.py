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

from . import qat  # noqa: F401
from .functional_layers import (  # noqa: F401
    FloatFunctionalLayer,
    add,
    concat,
    divide,
    flatten,
    matmul,
    multiply,
    reshape,
    subtract,
    transpose,
)
from .quant_layers import QuantStub  # noqa: F401
from .quantized_linear import (  # noqa: F401
    apply_per_channel_scale,
    llm_int8_linear,
    weight_dequantize,
    weight_only_linear,
    weight_quantize,
)
from .stub import Stub

__all__ = [
    "Stub",
    "weight_only_linear",
    "llm_int8_linear",
    "weight_quantize",
    "weight_dequantize",
]
