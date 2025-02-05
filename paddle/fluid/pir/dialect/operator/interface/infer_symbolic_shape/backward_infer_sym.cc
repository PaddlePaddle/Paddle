// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/backward_infer_sym.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"

namespace paddle::dialect {

bool FusedAttentionGradOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  bool is_test = op->attribute<pir::BoolAttribute>("is_test").data();
  PADDLE_ENFORCE_EQ(is_test,
                    false,
                    common::errors::InvalidArgument(
                        "GradOp is only callable when is_test is false"));
  bool pre_layer_norm =
      op->attribute<pir::BoolAttribute>("pre_layer_norm").data();
  auto same_shape_infer = [&](pir::Value &&dst, pir::Value &&src) {
    auto src_shape = infer_context->GetShapeOrDataForValue(src).shape();
    infer_context->SetShapeOrDataForValue(
        dst,
        symbol::ShapeOrDataDimExprs{
            symbol::TensorShapeOrDataDimExprs(src_shape)});
  };
  if (!pre_layer_norm) {
    if (!paddle::dialect::details::IsFakeValue(op->result(6)) &&
        op->operand_source(11)) {
      same_shape_infer(op->result(6), op->operand_source(11));
    }
    if (!paddle::dialect::details::IsFakeValue(op->result(7)) &&
        op->operand_source(12)) {
      same_shape_infer(op->result(7), op->operand_source(12));
    }
  }
  if (pre_layer_norm && op->operand_source(9)) {
    if (!paddle::dialect::details::IsFakeValue(op->result(4))) {
      same_shape_infer(op->result(4), op->operand_source(9));
    }
    if (!paddle::dialect::details::IsFakeValue(op->result(5)) &&
        op->operand_source(10)) {
      same_shape_infer(op->result(5), op->operand_source(10));
    }
  }
  same_shape_infer(op->result(8), op->operand_source(1));
  if (!paddle::dialect::details::IsFakeValue(op->result(3)) &&
      op->operand_source(13)) {
    same_shape_infer(op->result(3), op->operand_source(8));
  }
  if (!paddle::dialect::details::IsFakeValue(op->result(10))) {
    same_shape_infer(op->result(10), op->operand_source(7));
  }
  same_shape_infer(op->result(9), op->operand_source(2));
  if (!paddle::dialect::details::IsFakeValue(op->result(0))) {
    same_shape_infer(op->result(0), op->operand_source(3));
  }
  if (pre_layer_norm) {
    if (!paddle::dialect::details::IsFakeValue(op->result(11)) &&
        op->operand_source(13)) {
      same_shape_infer(op->result(11), op->operand_source(13));
    }
  } else {
    if (!paddle::dialect::details::IsFakeValue(op->result(12)) &&
        op->operand_source(18)) {
      same_shape_infer(op->result(12), op->operand_source(18));
    }
  }
  same_shape_infer(op->result(19), op->operand_source(26));
  same_shape_infer(op->result(14), op->operand_source(22));
  same_shape_infer(op->result(15), op->operand_source(20));
  same_shape_infer(op->result(16), op->operand_source(21));
  same_shape_infer(op->result(17), op->operand_source(23));
  same_shape_infer(op->result(18), op->operand_source(25));
  if (!paddle::dialect::details::IsFakeValue(op->result(2)) &&
      op->operand_source(6)) {
    same_shape_infer(op->result(2), op->operand_source(6));
  }
  same_shape_infer(op->result(13), op->operand_source(19));
  if (!paddle::dialect::details::IsFakeValue(op->result(1)) &&
      op->operand_source(4)) {
    same_shape_infer(op->result(1), op->operand_source(4));
  }
  same_shape_infer(op->result(20), op->operand_source(27));
  return true;
}
}  // namespace paddle::dialect
