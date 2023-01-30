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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from . import base
from .base import *

from . import layers
from .layers import *

<<<<<<< HEAD
=======
from . import container
from .container import *

from . import nn
from .nn import *

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from . import tracer
from .tracer import *

from . import parallel
from .parallel import *

<<<<<<< HEAD
=======
from . import checkpoint
from .checkpoint import *
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

from . import learning_rate_scheduler
from .learning_rate_scheduler import *

<<<<<<< HEAD
from .math_op_patch import monkey_patch_math_varbase

__all__ = []
__all__ += layers.__all__
__all__ += base.__all__
__all__ += parallel.__all__
__all__ += learning_rate_scheduler.__all__
=======
from . import jit
from .jit import *

from . import io
from .io import *

from . import static_runner
from .static_runner import StaticModelRunner

from . import dygraph_to_static
from .dygraph_to_static import ProgramTranslator

from . import rnn
from .rnn import *

from . import amp
from .amp import *

from .math_op_patch import monkey_patch_math_varbase

from .inplace_utils import inplace_apis_in_dygraph_only

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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
