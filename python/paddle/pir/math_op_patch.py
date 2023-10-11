# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from . import OpResult

_already_patch_opresult = False


def monkey_patch_opresult():
    global _already_patch_opresult
    if not _already_patch_opresult:
        # Handling Tensor Methods
        import paddle.tensor

        for method_name in paddle.tensor.tensor_method_func:
            if hasattr(OpResult, method_name):
                continue
            method_impl = getattr(paddle.tensor, method_name, None)
            if method_impl:
                setattr(OpResult, method_name, method_impl)

        # Handling __getitem__
        from ..base.variable_index import _getitem_static

        OpResult.__getitem__ = _getitem_static

        _already_patch_opresult = True
