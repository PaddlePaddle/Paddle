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

from .assert_transformer import AssertTransformer  # noqa: F401
from .ast_transformer import DygraphToStaticAst  # noqa: F401
from .convert_call_func import convert_call as Call  # noqa: F401
from .convert_operators import (  # noqa: F401
    convert_assert as Assert,
    convert_attr as Attr,
    convert_ifelse as IfElse,
    convert_len as Len,
    convert_load as Ld,
    convert_logical_and as And,
    convert_logical_not as Not,
    convert_logical_or as Or,
    convert_pop as Pop,
    convert_shape as Shape,
    convert_var_dtype as AsDtype,
    convert_while_loop as While,
    indexable as Indexable,
    unpack_by_structure as Unpack,
)
from .program_translator import convert_to_static  # noqa: F401
from .static_analysis import NodeVarType, StaticAnalysisVisitor  # noqa: F401
from .utils import UndefinedVar, ast_to_source_code, saw  # noqa: F401
from .variable_trans_func import (  # noqa: F401
    create_bool_as_type,
    to_static_variable,
)

__all__ = []
