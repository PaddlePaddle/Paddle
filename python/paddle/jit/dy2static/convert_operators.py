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

<<<<<<< HEAD
from ...fluid.dygraph.dygraph_to_static.convert_operators import (  # noqa: F401
    cast_bool_if_necessary,
    convert_assert,
    convert_ifelse,
    convert_len,
    convert_logical_and,
    convert_logical_not,
    convert_logical_or,
    convert_pop,
    convert_print,
    convert_shape_compare,
    convert_var_dtype,
    convert_shape,
    convert_while_loop,
    unpack_by_structure,
    indexable,
    convert_attr,
)
=======
from ...fluid.dygraph.dygraph_to_static.convert_operators import cast_bool_if_necessary  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_assert  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_ifelse  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_len  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_logical_and  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_logical_not  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_logical_or  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_pop  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_print  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_shape_compare  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_var_dtype  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_shape  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import convert_while_loop  # noqa: F401
from ...fluid.dygraph.dygraph_to_static.convert_operators import unpack_by_structure, indexable  # noqa: F401
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

__all__ = []
