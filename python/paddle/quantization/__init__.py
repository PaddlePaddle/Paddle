"""Quantization Module"""
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .imperative.ptq_config import (
    PTQConfig,
    default_ptq_config,
)
from .imperative.ptq_quantizer import (
    BaseQuantizer,
)
from .imperative.ptq_quantizer import (
    AbsmaxQuantizer,
)
from .imperative.ptq_quantizer import (
    PerChannelAbsmaxQuantizer,
)
from .imperative.ptq_quantizer import (
    KLQuantizer,
)
from .imperative.ptq_quantizer import (
    HistQuantizer,
)
from .imperative.ptq_quantizer import (
    SUPPORT_ACT_QUANTIZERS,
)
from .imperative.ptq_quantizer import (
    SUPPORT_WT_QUANTIZERS,
)
from .imperative.ptq_registry import (
    PTQRegistry,
)
from .imperative.ptq import (
    ImperativePTQ,
)
from .imperative.qat import (
    ImperativeQuantAware,
)

from .config import QuantConfig
from .base_quanter import BaseQuanter
from .base_observer import BaseObserver
from .factory import quanter
from .qat import QAT
from .ptq import PTQ

__all__ = [
    "QuantConfig",
    "BaseQuanter",
    "BaseObserver",
    "quanter",
    "QAT",
    "PTQ",
]
