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
from paddle import fluid  # noqa: F401
from paddle.fluid import core, dygraph_utils  # noqa: F401
from paddle.fluid.core import VarDesc  # noqa: F401
from paddle.fluid.data_feeder import (  # noqa: F401
    check_dtype,
    check_type,
    check_variable_and_dtype,
    convert_dtype,
)
from paddle.fluid.framework import (  # noqa: F401
    OpProtoHolder,
    Variable,
    _create_tensor,
    _dygraph_tracer,
    convert_np_dtype_to_dtype_,
    default_main_program,
    device_guard,
    dygraph_only,
    in_dygraph_mode,
)
from paddle.fluid.layer_helper import LayerHelper  # noqa: F401
from paddle.fluid.layers.layer_function_generator import (  # noqa: F401
    templatedoc,
)
from paddle.fluid.param_attr import ParamAttr  # noqa: F401
