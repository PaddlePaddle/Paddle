#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from . import (  # noqa: F401
    amp_lists,
    amp_utils,
    decorator,
)
from .amp_lists import AutoMixedPrecisionListsBF16  # noqa: F401
from .amp_utils import (  # noqa: F401
    bf16_guard,
    cast_model_to_bf16,
    cast_parameters_to_bf16,
    convert_float_to_uint16,
    rewrite_program_bf16,
)
from .decorator import decorate_bf16  # noqa: F401
