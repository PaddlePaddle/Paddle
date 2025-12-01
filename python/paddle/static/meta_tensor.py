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

from ..base.libpaddle import IrMetaTensor, IrTensor


class MetaTensor:
    def __init__(self, shape=[], dtype="float32"):
        self.ir_tensor = IrTensor()
        self.ir_meta_tensor = IrMetaTensor(self.ir_tensor)
        self.ir_meta_tensor.set_shape(shape)
        self.ir_meta_tensor.set_dtype(dtype)

    def set_shape(self, shape):
        self.ir_meta_tensor.set_shape(shape)

    @property
    def shape(self):
        return self.ir_meta_tensor.shape

    def set_dtype(self, dtype):
        self.ir_meta_tensor.set_dtype(dtype)

    @property
    def dtype(self):
        return self.ir_meta_tensor.dtype

    def __eq__(self, other):
        return (
            self.ir_meta_tensor.dtype == other.ir_meta_tensor.dtype
            and self.ir_meta_tensor.shape == other.ir_meta_tensor.shape
        )
