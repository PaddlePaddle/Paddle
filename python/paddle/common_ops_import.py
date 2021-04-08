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
from six.moves import reduce  # noqa: F401
from paddle.fluid.layer_helper import LayerHelper  # noqa: F401
from paddle.fluid.param_attr import ParamAttr  # noqa: F401
from paddle.fluid.framework import convert_np_dtype_to_dtype_  # noqa: F401
from paddle.fluid.framework import in_dygraph_mode  # noqa: F401
from paddle.fluid.framework import _varbase_creator  # noqa: F401
from paddle.fluid.framework import device_guard  # noqa: F401
from paddle.fluid.framework import default_main_program  # noqa: F401
from paddle.fluid.framework import dygraph_only  # noqa: F401
from paddle.fluid.framework import _dygraph_tracer  # noqa: F401
from paddle.fluid.framework import OpProtoHolder  # noqa: F401
from paddle.fluid.framework import Variable  # noqa: F401
from paddle.fluid.initializer import Constant  # noqa: F401
from paddle.fluid.core import VarDesc  # noqa: F401
from paddle.fluid import core  # noqa: F401
from paddle.fluid import dygraph_utils  # noqa: F401
from paddle.fluid.data_feeder import check_type  # noqa: F401
from paddle.fluid.data_feeder import check_dtype  # noqa: F401
from paddle.fluid.data_feeder import check_variable_and_dtype  # noqa: F401
from paddle.fluid.data_feeder import convert_dtype  # noqa: F401
from paddle.fluid.layers import fill_constant  # noqa: F401
from paddle.fluid.layers import utils  # noqa: F401
from paddle.fluid.layers import scale  # noqa: F401
from paddle.fluid.layers.layer_function_generator import templatedoc  # noqa: F401
import paddle.fluid as fluid  # noqa: F401
