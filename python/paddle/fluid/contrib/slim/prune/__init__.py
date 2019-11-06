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

from . import pruner
from .pruner import *
from . import prune_strategy
from .prune_strategy import *
from . import auto_prune_strategy
from .auto_prune_strategy import *

__all__ = pruner.__all__
__all__ += prune_strategy.__all__
__all__ += auto_prune_strategy.__all__
