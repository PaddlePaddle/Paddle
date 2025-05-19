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

from .facade_op import FacadeOp as FacadeOp


class TupleIdentityOp(FacadeOp):
    def __init__(self):
        super().__init__()

    def custom_op_name(self) -> str:
        return "ap_custom_op.tuple_identity"

    def infer_meta(self) -> str:
        return "facade_utils.tuple_identity_infer_meta"

    def infer_symbolic(self) -> str:
        return "facade_utils.tuple_identity_infer_symbolic"

    def num_inputs(self) -> int:
        return -1

    def num_outputs(self, args) -> int:
        # the same as inputs
        return len(args)

    def attributes_schema(self):
        # annotations matter.
        pass


TieOp = TupleIdentityOp


class FacadeQuantOp(FacadeOp):
    def __init__(self):
        super().__init__()

    def custom_op_name(self) -> str:
        return "ap_custom_op.facade_quant"

    def infer_meta(self) -> str:
        return "facade_utils.quant_infer_meta"

    def infer_symbolic(self) -> str:
        return "facade_utils.quant_infer_symbolic"

    def num_inputs(self) -> int:
        return 1

    def num_outputs(self, args) -> int:
        return 2

    def attributes_schema(self):
        # annotations matter.
        pass
