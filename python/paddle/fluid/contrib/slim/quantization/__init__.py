# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import quantization_pass
from .quantization_pass import *
from . import quantization_strategy
from .quantization_strategy import *
from . import mkldnn_post_training_strategy
from .mkldnn_post_training_strategy import *
from . import quantization_mkldnn_pass
from .quantization_mkldnn_pass import *

__all__ = quantization_pass.__all__ + quantization_strategy.__all__
__all__ += mkldnn_post_training_strategy.__all__
__all__ += quantization_mkldnn_pass.__all__
