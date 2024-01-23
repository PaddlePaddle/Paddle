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

from paddle.base.core import (
    InternalUtils,  # noqa: F401
    PredictorPool,
    XpuConfig,
    _get_phi_kernel_name,
    create_predictor,
    get_num_bytes_of_data_type,
    get_trt_compile_version,
    get_trt_runtime_version,
    get_version,
)

from .wrapper import (
    Config,
    DataType,
    PlaceType,
    PrecisionType,
    Predictor,
    Tensor,
    convert_to_mixed_precision,
)

__all__ = [
    'Config',
    'DataType',
    'PlaceType',
    'PrecisionType',
    'Tensor',
    'Predictor',
    'create_predictor',
    'get_version',
    '_get_phi_kernel_name',
    'get_trt_compile_version',
    'convert_to_mixed_precision',
    'get_trt_runtime_version',
    'get_num_bytes_of_data_type',
    'PredictorPool',
    'XpuConfig',
]
