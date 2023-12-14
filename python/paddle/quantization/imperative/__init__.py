#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from . import (
    ptq,  # noqa: F401
    ptq_config,  # noqa: F401
    ptq_quantizer,  # noqa: F401
    ptq_registry,  # noqa: F401
    qat,  # noqa: F401
)
from .ptq import ImperativePTQ  # noqa: F401
from .ptq_config import PTQConfig, default_ptq_config  # noqa: F401
from .ptq_quantizer import (  # noqa: F401
    SUPPORT_ACT_QUANTIZERS,
    SUPPORT_WT_QUANTIZERS,
    AbsmaxQuantizer,
    BaseQuantizer,
    HistQuantizer,
    KLQuantizer,
    PerChannelAbsmaxQuantizer,
)
from .ptq_registry import PTQRegistry  # noqa: F401
from .qat import ImperativeQuantAware  # noqa: F401
