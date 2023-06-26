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

from .utils import saw, UndefinedVar, ast_to_source_code
from .convert_operators import convert_logical_and as And  # noqa: F401
from .convert_operators import convert_var_dtype as AsDtype  # noqa: F401
from .convert_operators import convert_assert as Assert  # noqa: F401
from .convert_call_func import convert_call as Call  # noqa: F401
from .convert_operators import convert_ifelse as IfElse  # noqa: F401
from .convert_operators import convert_len as Len  # noqa: F401
from .convert_operators import convert_logical_not as Not  # noqa: F401
from .convert_operators import convert_logical_or as Or  # noqa: F401
from .convert_operators import convert_pop as Pop  # noqa: F401
from .convert_operators import convert_shape as Shape  # noqa: F401
from .convert_operators import convert_while_loop as While  # noqa: F401
from .convert_operators import unpack_by_structure as Unpack  # noqa: F401
from .convert_operators import convert_attr as Attr  # noqa: F401
from .convert_operators import convert_load as Ld  # noqa: F401
from .convert_operators import indexable as Indexable  # noqa: F401
from .variable_trans_func import create_bool_as_type  # noqa: F401
from .variable_trans_func import to_static_variable  # noqa: F401
from .convert_operators import convert_shape_compare  # noqa: F401
from .assert_transformer import AssertTransformer
from .ast_transformer import DygraphToStaticAst
from .program_translator import convert_to_static
from .static_analysis import NodeVarType, StaticAnalysisVisitor

__all__ = []
