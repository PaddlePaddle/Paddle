# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
common import
"""
from six.moves import reduce
from fluid.layer_helper import LayerHelper
from fluid.param_attr import ParamAttr
from fluid.framework import convert_np_dtype_to_dtype_, in_dygraph_mode, _varbase_creator
from fluid.framework import Variable
from fluid.initializer import Constant
from fluid.core import VarDesc
from fluid import core
from fluid.data_feeder import check_type, check_dtype, convert_dtype
from fluid.layers import utils
import numpy
import warnings
