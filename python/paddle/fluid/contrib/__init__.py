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
from .decoder import *
from . import memory_usage_calc
from .memory_usage_calc import *
from . import op_frequence
from .op_frequence import *
from . import quantize
from .quantize import *
from . import int8_inference
from .int8_inference import *
from . import reader
from .reader import *
from . import slim
from .slim import *
from . import utils
from .utils import *

__all__ = []
__all__ += decoder.__all__
__all__ += memory_usage_calc.__all__
__all__ += op_frequence.__all__
__all__ += quantize.__all__
__all__ += int8_inference.__all__
__all__ += reader.__all__
__all__ += slim.__all__
__all__ += utils.__all__
