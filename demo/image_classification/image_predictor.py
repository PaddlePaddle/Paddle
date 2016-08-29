# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

import os
import numpy as np
from optparse import OptionParser

from py_paddle import swig_paddle, util, DataProviderWrapperConverter
from paddle.trainer.PyDataProviderWrapper import DenseSlot
from paddle.trainer.config_parser import parse_config



"""
Will merge predictor from Qingqing.
"""
