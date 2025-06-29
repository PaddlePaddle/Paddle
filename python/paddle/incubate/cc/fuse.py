# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from contextlib import contextmanager

import paddle

__all__ = ['matmul', 'by_register']


@contextmanager
def by_register():
    paddle._C_ops.ap_trivial_fusion_begin(None)
    yield
    paddle._C_ops.ap_trivial_fusion_end(None)


def matmul(x, w, epilogue, **kwargs):
    x = paddle.matmul(x, w, **kwargs)
    with by_register():
        return epilogue(x)
