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
from paddle.fluid.layer_helper import LayerHelper  # noqa: F401
from paddle.fluid.param_attr import ParamAttr  # noqa: F401
from paddle.fluid.framework import (  # noqa: F401
    convert_np_dtype_to_dtype_,
    _non_static_mode,
    _varbase_creator,
    in_dygraph_mode,
    _in_legacy_dygraph,
)
from paddle.fluid.framework import (  # noqa: F401
    device_guard,
    default_main_program,
    dygraph_only,
    _dygraph_tracer,
)
from paddle.fluid.framework import OpProtoHolder, Variable  # noqa: F401
from paddle.fluid.initializer import Constant  # noqa: F401
from paddle.fluid.core import VarDesc  # noqa: F401
from paddle.fluid import core, dygraph_utils  # noqa: F401
from paddle.fluid.data_feeder import (  # noqa: F401
    check_type,
    check_dtype,
    check_variable_and_dtype,
    convert_dtype,
)
from paddle.fluid.layers import fill_constant, utils, scale  # noqa: F401
from paddle.fluid.layers.layer_function_generator import (  # noqa: F401
    templatedoc,
)
import paddle.fluid as fluid  # noqa: F401
