// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/ap_infer_sym.h"
#ifdef PADDLE_WITH_CINN
#include "paddle/ap/include/paddle/pir/infer_symbolic_shape_util.h"
#endif
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

namespace paddle::dialect {

bool ApTrivialFusionBeginOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
#ifdef PADDLE_WITH_CINN
  symbol::ShapeOrDataDimExprs empty_shape{
      symbol::TensorShapeOrDataDimExprs{std::vector<symbol::DimExpr>{}}};
  infer_context->SetShapeOrDataForValue(op->result(0), empty_shape);
  return true;
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "ap_trivial_fusion_begin is not implemented when cinn is not enabled."));
  return false;
#endif
}

bool ApTrivialFusionEndOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
#ifdef PADDLE_WITH_CINN
  symbol::ShapeOrDataDimExprs empty_shape{
      symbol::TensorShapeOrDataDimExprs{std::vector<symbol::DimExpr>{}}};
  infer_context->SetShapeOrDataForValue(op->result(0), empty_shape);
  return true;
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "ap_trivial_fusion_end is not implemented when cinn is not enabled."));
  return false;
#endif
}

bool ApFacadeOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
#ifdef PADDLE_WITH_CINN
  return ap::dialect::PdOpApFacadeOpInferSymbolicShape(op, infer_context);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "ap_facade is not implemented when cinn is not enabled."));
  return false;
#endif
}

bool ApVariadicOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
#ifdef PADDLE_WITH_CINN
  return ap::dialect::PdOpApVariadicOpInferSymbolicShape(op, infer_context);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "ap_variadic is not implemented when cinn is not enabled."));
  return false;
#endif
}

}  // namespace paddle::dialect
