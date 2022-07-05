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

from paddle.utils import gast

# Repeat the definition here to solve the problem of circular import
ORIGI_INFO = "Original information of source code for ast node."


class BaseTransformer(gast.NodeTransformer):

    def visit(self, node):
        if not isinstance(node, gast.AST):
            msg = ('Expected "gast.AST", but got "{}".').format(type(node))
            raise ValueError(msg)
        origin_info = getattr(node, ORIGI_INFO, None)

        result = super(BaseTransformer, self).visit(node)

        iter_result = result
        if iter_result is not node and iter_result is not None:
            if not isinstance(iter_result, (list, tuple)):
                iter_result = (iter_result, )
            if origin_info is not None:
                for n in iter_result:
                    setattr(n, ORIGI_INFO, origin_info)

        return result
