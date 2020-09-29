#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function

from . import decoder
from . import extend_optimizer
from . import layers
from . import memory_usage_calc
from . import mixed_precision
from . import model_stat
from . import op_frequence
from . import quantize
from . import reader
from . import slim
from . import utils
from .decoder import *
from .extend_optimizer import *
from .layers import *
from .memory_usage_calc import *
from .mixed_precision import *
from .model_stat import *
from .op_frequence import *
from .quantize import *
from .reader import *
from .utils import *

__all__ = []
__all__ += decoder.__all__
__all__ += memory_usage_calc.__all__
__all__ += op_frequence.__all__
__all__ += quantize.__all__
__all__ += reader.__all__
__all__ += utils.__all__
__all__ += extend_optimizer.__all__
__all__ += ['mixed_precision']
__all__ += layers.__all__
