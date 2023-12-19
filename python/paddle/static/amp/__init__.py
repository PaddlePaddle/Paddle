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

from . import (  # noqa: F401
    bf16,
    debugging,
    decorator,
    fp16_lists,
    fp16_utils,
)
from .decorator import decorate  # noqa: F401
from .fp16_lists import AutoMixedPrecisionLists, CustomOpLists  # noqa: F401
from .fp16_utils import (  # noqa: F401
    cast_model_to_fp16,
    cast_parameters_to_fp16,
    fp16_guard,
)
