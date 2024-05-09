// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/same_operands_result.h"

#define OP_SAME_OPERANDS_AND_RESULT(name)                                  \
  bool name##OpInferSymbolicShape(                                         \
      pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) { \
    const symbol::ShapeOrDataDimExprs &operand_shape_or_data =             \
        infer_context->GetShapeOrDataForValue(op->operand_source(0));      \
    infer_context->SetShapeOrDataForValue(op->result(0),                   \
                                          operand_shape_or_data);          \
    return true;                                                           \
  }

namespace paddle::dialect {

OP_SAME_OPERANDS_AND_RESULT(Abs)
OP_SAME_OPERANDS_AND_RESULT(Abs_)
OP_SAME_OPERANDS_AND_RESULT(Acos)
OP_SAME_OPERANDS_AND_RESULT(Acos_)
OP_SAME_OPERANDS_AND_RESULT(Acosh)
OP_SAME_OPERANDS_AND_RESULT(Acosh_)
OP_SAME_OPERANDS_AND_RESULT(Angle)
OP_SAME_OPERANDS_AND_RESULT(Argsort)
OP_SAME_OPERANDS_AND_RESULT(Asin)
OP_SAME_OPERANDS_AND_RESULT(Asin_)
OP_SAME_OPERANDS_AND_RESULT(Asinh)
OP_SAME_OPERANDS_AND_RESULT(Asinh_)
OP_SAME_OPERANDS_AND_RESULT(Assign)
OP_SAME_OPERANDS_AND_RESULT(Assign_)
OP_SAME_OPERANDS_AND_RESULT(Atan)
OP_SAME_OPERANDS_AND_RESULT(Atan_)
OP_SAME_OPERANDS_AND_RESULT(Atanh)
OP_SAME_OPERANDS_AND_RESULT(Atanh_)
OP_SAME_OPERANDS_AND_RESULT(Bernoulli)
OP_SAME_OPERANDS_AND_RESULT(BitwiseNot)
OP_SAME_OPERANDS_AND_RESULT(BitwiseNot_)
OP_SAME_OPERANDS_AND_RESULT(Cast)
OP_SAME_OPERANDS_AND_RESULT(Cast_)
OP_SAME_OPERANDS_AND_RESULT(Ceil)
OP_SAME_OPERANDS_AND_RESULT(Ceil_)
OP_SAME_OPERANDS_AND_RESULT(Conj)
OP_SAME_OPERANDS_AND_RESULT(Cos)
OP_SAME_OPERANDS_AND_RESULT(Cos_)
OP_SAME_OPERANDS_AND_RESULT(Cosh)
OP_SAME_OPERANDS_AND_RESULT(Cosh_)
OP_SAME_OPERANDS_AND_RESULT(Digamma)
OP_SAME_OPERANDS_AND_RESULT(Digamma_)
OP_SAME_OPERANDS_AND_RESULT(Dirichlet)
OP_SAME_OPERANDS_AND_RESULT(Equal)
OP_SAME_OPERANDS_AND_RESULT(Equal_)
OP_SAME_OPERANDS_AND_RESULT(Erf)
OP_SAME_OPERANDS_AND_RESULT(Erf_)
OP_SAME_OPERANDS_AND_RESULT(Erfinv)
OP_SAME_OPERANDS_AND_RESULT(Erfinv_)
OP_SAME_OPERANDS_AND_RESULT(Exp)
OP_SAME_OPERANDS_AND_RESULT(Exp_)
OP_SAME_OPERANDS_AND_RESULT(Expm1)
OP_SAME_OPERANDS_AND_RESULT(Expm1_)
OP_SAME_OPERANDS_AND_RESULT(Exponential_)
OP_SAME_OPERANDS_AND_RESULT(Fetch)
OP_SAME_OPERANDS_AND_RESULT(Flip)
OP_SAME_OPERANDS_AND_RESULT(Floor)
OP_SAME_OPERANDS_AND_RESULT(Floor_)
OP_SAME_OPERANDS_AND_RESULT(Imag)
OP_SAME_OPERANDS_AND_RESULT(Increment)
OP_SAME_OPERANDS_AND_RESULT(Increment_)
OP_SAME_OPERANDS_AND_RESULT(Isfinite)
OP_SAME_OPERANDS_AND_RESULT(IsfiniteSr)
OP_SAME_OPERANDS_AND_RESULT(Isinf)
OP_SAME_OPERANDS_AND_RESULT(IsinfSr)
OP_SAME_OPERANDS_AND_RESULT(Isnan)
OP_SAME_OPERANDS_AND_RESULT(IsnanSr)
OP_SAME_OPERANDS_AND_RESULT(Lgamma)
OP_SAME_OPERANDS_AND_RESULT(Lgamma_)
OP_SAME_OPERANDS_AND_RESULT(Log1p)
OP_SAME_OPERANDS_AND_RESULT(Log1p_)
OP_SAME_OPERANDS_AND_RESULT(Log)
OP_SAME_OPERANDS_AND_RESULT(Log_)
OP_SAME_OPERANDS_AND_RESULT(LogicalNot)
OP_SAME_OPERANDS_AND_RESULT(LogicalNot_)
OP_SAME_OPERANDS_AND_RESULT(Logit)
OP_SAME_OPERANDS_AND_RESULT(Logit_)
OP_SAME_OPERANDS_AND_RESULT(Pow)
OP_SAME_OPERANDS_AND_RESULT(Poisson)
OP_SAME_OPERANDS_AND_RESULT(Pow_)
OP_SAME_OPERANDS_AND_RESULT(Print)
OP_SAME_OPERANDS_AND_RESULT(PutAlongAxis)
OP_SAME_OPERANDS_AND_RESULT(PutAlongAxis_)
OP_SAME_OPERANDS_AND_RESULT(Real)
OP_SAME_OPERANDS_AND_RESULT(Reciprocal)
OP_SAME_OPERANDS_AND_RESULT(Reciprocal_)
OP_SAME_OPERANDS_AND_RESULT(Relu)
OP_SAME_OPERANDS_AND_RESULT(Relu6)
OP_SAME_OPERANDS_AND_RESULT(Relu_)
OP_SAME_OPERANDS_AND_RESULT(Reverse)
OP_SAME_OPERANDS_AND_RESULT(Roll)
OP_SAME_OPERANDS_AND_RESULT(Round)
OP_SAME_OPERANDS_AND_RESULT(Round_)
OP_SAME_OPERANDS_AND_RESULT(Rsqrt)
OP_SAME_OPERANDS_AND_RESULT(Rsqrt_)
OP_SAME_OPERANDS_AND_RESULT(ScaleSr)
OP_SAME_OPERANDS_AND_RESULT(ScaleSr_)
OP_SAME_OPERANDS_AND_RESULT(Scale_)
OP_SAME_OPERANDS_AND_RESULT(ScatterNdAdd)
OP_SAME_OPERANDS_AND_RESULT(Scatter)
OP_SAME_OPERANDS_AND_RESULT(Scatter_)
OP_SAME_OPERANDS_AND_RESULT(Select)
OP_SAME_OPERANDS_AND_RESULT(Sign)
OP_SAME_OPERANDS_AND_RESULT(Sin)
OP_SAME_OPERANDS_AND_RESULT(Sin_)
OP_SAME_OPERANDS_AND_RESULT(Sinh)
OP_SAME_OPERANDS_AND_RESULT(Sinh_)
OP_SAME_OPERANDS_AND_RESULT(Softmax)
OP_SAME_OPERANDS_AND_RESULT(Softmax_)
OP_SAME_OPERANDS_AND_RESULT(Tan)
OP_SAME_OPERANDS_AND_RESULT(Tan_)
OP_SAME_OPERANDS_AND_RESULT(Tanh)
OP_SAME_OPERANDS_AND_RESULT(Tanh_)
OP_SAME_OPERANDS_AND_RESULT(Tril)
OP_SAME_OPERANDS_AND_RESULT(Tril_)
OP_SAME_OPERANDS_AND_RESULT(Triu)
OP_SAME_OPERANDS_AND_RESULT(Triu_)
OP_SAME_OPERANDS_AND_RESULT(Trunc)
OP_SAME_OPERANDS_AND_RESULT(Trunc_)
OP_SAME_OPERANDS_AND_RESULT(Sigmoid)
OP_SAME_OPERANDS_AND_RESULT(Sigmoid_)

bool ScaleOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  std::vector<symbol::DimExpr> shape(operand_shape_or_data.shape());

  if (operand_shape_or_data.data()) {
    const std::vector<symbol::DimExpr> data = [&] {
      const symbol::DimExpr scale = [&]() -> symbol::DimExpr {
        if (op->num_operands() == 2) {
          return infer_context->GetShapeOrDataForValue(op->operand_source(1))
              .data()
              ->at(0);
        }
        return static_cast<int64_t>(
            op->attribute("scale").dyn_cast<pir::FloatAttribute>().data());
      }();
      int bias = op->attribute("bias").dyn_cast<pir::FloatAttribute>().data();

      std::vector<symbol::DimExpr> data;
      for (auto &val : *(operand_shape_or_data.data())) {
        data.push_back(val * scale + bias);
      }
      return data;
    }();

    infer_context->SetShapeOrDataForValue(
        op->result(0), symbol::TensorShapeOrDataDimExprs(shape, data));
  } else {
    infer_context->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  }

  return true;
}

}  // namespace paddle::dialect

namespace cinn::dialect {
using paddle::dialect::ReverseOpInferSymbolicShape;
using paddle::dialect::ScaleOpInferSymbolicShape;
}  // namespace cinn::dialect

#undef OP_SAME_OPERANDS_AND_RESULT
