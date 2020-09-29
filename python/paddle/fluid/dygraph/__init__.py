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

from . import amp
from . import base
from . import checkpoint
from . import container
from . import dygraph_to_static
from . import io
from . import jit
from . import layers
from . import learning_rate_scheduler
from . import nn
from . import parallel
from . import rnn
from . import static_runner
from . import tracer
from .amp import *
from .base import *
from .checkpoint import *
from .container import *
from .dygraph_to_static import ProgramTranslator
from .io import *
from .jit import *
from .layers import *
from .learning_rate_scheduler import *
from .math_op_patch import monkey_patch_math_varbase
from .nn import *
from .parallel import *
from .rnn import *
from .static_runner import StaticModelRunner
from .tracer import *

__all__ = []
__all__ += layers.__all__
__all__ += base.__all__
__all__ += container.__all__
__all__ += nn.__all__
__all__ += parallel.__all__
__all__ += checkpoint.__all__
__all__ += learning_rate_scheduler.__all__
__all__ += jit.__all__
__all__ += io.__all__
__all__ += rnn.__all__
__all__ += ['ProgramTranslator']
__all__ += amp.__all__
