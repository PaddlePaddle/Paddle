#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ...fluid.dygraph.dygraph_to_static.convert_operators import cast_bool_if_necessary  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_assert  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_ifelse  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_len  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_logical_and  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_logical_not  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_logical_or  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_pop  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_print  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_shape_compare  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_var_dtype  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_var_shape  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_var_shape_simple  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import eval_if_exist_else_none  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import choose_shape_attr_or_api  #DEFINE_ALIAS
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_while_loop  #DEFINE_ALIAS

__all__ = [
    'cast_bool_if_necessary', 'convert_assert', 'convert_ifelse', 'convert_len',
    'convert_logical_and', 'convert_logical_not', 'convert_logical_or',
    'convert_pop', 'convert_print', 'convert_shape_compare',
    'convert_var_dtype', 'convert_var_shape', 'convert_var_shape_simple',
    'eval_if_exist_else_none', 'choose_shape_attr_or_api', 'convert_while_loop'
]
