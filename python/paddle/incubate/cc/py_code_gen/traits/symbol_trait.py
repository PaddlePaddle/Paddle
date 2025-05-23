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

from paddle.incubate.cc.py_code_gen.ir import ir_symbol


class SymbolTrait:

    def s_int64(self, value):
        if value < 0:
            return self.s_negative(self.s_int64(-value))
        return ir_symbol.Int64(value)

    def s_str(self, value):
        return ir_symbol.String(value)

    def s_negative(self, value):
        return ir_symbol.Negative(value)

    def s_reciprocal(self, value):
        return ir_symbol.Reciprocal(value)

    def s_add(self, *args):
        return ir_symbol.Add(args)

    def s_mul(self, *args):
        return ir_symbol.Mul(args)

    def s_max(self, *args):
        return ir_symbol.Max(args)

    def s_min(self, *args):
        return ir_symbol.Min(args)

    def s_broadcast(self, *args):
        return ir_symbol.Broadcast(args)

    def s_null(self):
        return ir_symbol.NullShapeOrDataDimExprs()

    def s_tensor_shape_or_data(self, shape, data):
        return ir_symbol.TensorShapeOrDataDimExprs(shape=shape, data=data)

    def s_tensor_list_shape_or_data(self, *args):
        return ir_symbol.TensorListShapeOrDataDimExprs(args)
