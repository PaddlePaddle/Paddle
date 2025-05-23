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

from paddle.incubate.cc.py_code_gen.ir import ir_tensor, ir_type
from paddle.incubate.cc.py_code_gen.ir_converters import paddle_type_converter


def ConvertToPaddleTensor(tensor):
    return getattr(PaddleTensorConverter, type(tensor.type).__name__)(tensor)


class PaddleTensorConverter:

    @classmethod
    def DenseTensorType(cls, tensor):
        return ir_tensor.Tensor(
            local_name_prefix=tensor.local_name_prefix,
            name=tensor.name,
            arg_name_as_input=tensor.arg_name_as_input,
            defining_op_name=tensor.defining_op_name,
            type=ir_type.DenseTensorType(
                tensor.shape,
                paddle_type_converter.ConvertTypeToString(tensor.dtype),
            ),
            dim_exprs=tensor.dim_exprs,
        )

    @classmethod
    def VectorType(cls, tensor):
        return ir_tensor.Tensor(
            local_name_prefix=tensor.local_name_prefix,
            name=tensor.name,
            arg_name_as_input=tensor.arg_name_as_input,
            defining_op_name=tensor.defining_op_name,
            type=ir_type.VectorType(
                value=[
                    ir_type.DenseTensorType(
                        t.shape,
                        paddle_type_converter.ConvertTypeToString(t.dtype),
                    )
                    for t in tensor.type.value
                ]
            ),
            dim_exprs=tensor.dim_exprs,
        )

    @classmethod
    def NullType(cls, tensor):
        return ir_tensor.Tensor(
            local_name_prefix=tensor.local_name_prefix,
            name=tensor.name,
            arg_name_as_input=tensor.arg_name_as_input,
            defining_op_name=tensor.defining_op_name,
            type=tensor.type,
            dim_exprs=tensor.dim_exprs,
        )
