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

from . import config
from .config import *
from . import compress_pass
from .compress_pass import *
from . import strategy
from .strategy import *
from . import pass_builder
from .pass_builder import *

__all__ = config.__all__ + compress_pass.__all__ + strategy.__all__ + pass_builder.__all__
