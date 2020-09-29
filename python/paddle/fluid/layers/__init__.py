#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from . import control_flow
from . import detection
from . import device
from . import io
from . import loss
from . import math_op_patch
from . import metric_op
from . import nn
from . import ops
from . import rnn
from . import tensor
from .collective import *
from .control_flow import *
from .detection import *
from .device import *
from .distributions import *
from .io import *
from .learning_rate_scheduler import *
from .loss import *
from .math_op_patch import *
from .metric_op import *
from .nn import *
from .ops import *
from .sequence_lod import *
from .tensor import *

__all__ = []
__all__ += nn.__all__
__all__ += io.__all__
__all__ += tensor.__all__
__all__ += control_flow.__all__
__all__ += ops.__all__
__all__ += device.__all__
__all__ += detection.__all__
__all__ += metric_op.__all__
__all__ += learning_rate_scheduler.__all__
__all__ += distributions.__all__
__all__ += sequence_lod.__all__
__all__ += loss.__all__
__all__ += rnn.__all__

from .rnn import *
