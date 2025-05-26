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

import warnings

import paddle

from .pir_attrs_serializer import PirAttrsSerializer


class FacadeOp:

    def __init__(self):
        self.custom_op_name_ = self.custom_op_name()
        self.infer_meta_ = self._check_to_str_pair(self.infer_meta())
        self.infer_symbolic_ = self._check_to_str_pair(self.infer_symbolic())
        self.num_inputs_ = self.num_inputs()
        self.attrs_serializer_ = PirAttrsSerializer(self.attributes_schema)

    def custom_op_name(self) -> str:
        raise NotImplementedError(
            "static method custom_op_name() is not overwritten"
        )

    def infer_meta(self) -> str:
        raise NotImplementedError(
            "static method infer_meta() is not overwritten"
        )

    def infer_symbolic(self) -> str:
        raise NotImplementedError(
            "static method infer_symbolic() is not overwritten"
        )

    def num_inputs(self) -> int:
        raise NotImplementedError(
            "static method num_inputs() is not overwritten"
        )

    def num_outputs(self, args) -> int:
        raise NotImplementedError(
            "static method num_outputs() is not overwritten"
        )

    def attributes_schema(self):
        # annotations matter.
        raise NotImplementedError(
            "static method attributes_schema() is not overwritten"
        )

    def __call__(self, args, **kwargs):
        if paddle.in_dynamic_mode():
            warnings.warn("ap FacadeOp should not run in dynamic mode")
        assert isinstance(args, (tuple, list))
        self._check_num_inputs(len(args))
        serialized_attrs = self.attrs_serializer_(**kwargs)
        ret = paddle._C_ops.ap_facade(
            args if len(args) > 0 else None,
            self.num_outputs(args),
            self.custom_op_name_,
            self.infer_meta_,
            self.infer_symbolic_,
            serialized_attrs,
        )
        self._check_num_outputs(args, len(ret))
        return ret

    def _check_num_inputs(self, num_args):
        if self.num_inputs_ >= 0:
            assert self.num_inputs_ == num_args

    def _check_num_outputs(self, args, num_rets):
        num_outputs = self.num_outputs(args)
        if num_outputs >= 0:
            assert num_outputs == num_rets

    def _check_to_str_pair(self, pair_str):
        assert isinstance(pair_str, str)
        pair = pair_str.split(".")
        assert len(pair) == 2
        assert pair[0] not in (None, "")
        assert pair[1] not in (None, "")
        return pair_str
