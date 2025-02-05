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
#include <optional>

#define OP_SAME_OPERANDS_AND_RESULT(name)                                     \
  bool name##OpInferSymbolicShape(                                            \
      pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {    \
    const auto &operand_shape =                                               \
        infer_context->GetShapeOrDataForValue(op->operand_source(0)).shape(); \
    infer_context->SetShapeOrDataForValue(                                    \
        op->result(0),                                                        \
        symbol::ShapeOrDataDimExprs{                                          \
            symbol::TensorShapeOrDataDimExprs(operand_shape)});               \
    return true;                                                              \
  }

namespace paddle::dialect {

OP_SAME_OPERANDS_AND_RESULT(Abs)
OP_SAME_OPERANDS_AND_RESULT(Abs_)
OP_SAME_OPERANDS_AND_RESULT(Acos)
OP_SAME_OPERANDS_AND_RESULT(Acos_)
OP_SAME_OPERANDS_AND_RESULT(Acosh)
OP_SAME_OPERANDS_AND_RESULT(Acosh_)
OP_SAME_OPERANDS_AND_RESULT(Angle)
OP_SAME_OPERANDS_AND_RESULT(Asin)
OP_SAME_OPERANDS_AND_RESULT(Asin_)
OP_SAME_OPERANDS_AND_RESULT(Asinh)
OP_SAME_OPERANDS_AND_RESULT(Asinh_)
OP_SAME_OPERANDS_AND_RESULT(Atan)
OP_SAME_OPERANDS_AND_RESULT(Atan_)
OP_SAME_OPERANDS_AND_RESULT(Atanh)
OP_SAME_OPERANDS_AND_RESULT(Atanh_)
OP_SAME_OPERANDS_AND_RESULT(Hardtanh)
OP_SAME_OPERANDS_AND_RESULT(Hardtanh_)
OP_SAME_OPERANDS_AND_RESULT(Bernoulli)
OP_SAME_OPERANDS_AND_RESULT(BitwiseNot)
OP_SAME_OPERANDS_AND_RESULT(BitwiseNot_)
OP_SAME_OPERANDS_AND_RESULT(Celu)
OP_SAME_OPERANDS_AND_RESULT(Clip)
OP_SAME_OPERANDS_AND_RESULT(Clip_)
OP_SAME_OPERANDS_AND_RESULT(Conj)
OP_SAME_OPERANDS_AND_RESULT(CopyTo)
OP_SAME_OPERANDS_AND_RESULT(Cos)
OP_SAME_OPERANDS_AND_RESULT(Cos_)
OP_SAME_OPERANDS_AND_RESULT(Cosh)
OP_SAME_OPERANDS_AND_RESULT(Cosh_)
OP_SAME_OPERANDS_AND_RESULT(DequantizeAbsMax)
OP_SAME_OPERANDS_AND_RESULT(DequantizeLog)
OP_SAME_OPERANDS_AND_RESULT(Digamma)
OP_SAME_OPERANDS_AND_RESULT(Digamma_)
OP_SAME_OPERANDS_AND_RESULT(Dirichlet)
OP_SAME_OPERANDS_AND_RESULT(DisableCheckModelNanInf)
OP_SAME_OPERANDS_AND_RESULT(Depend)
OP_SAME_OPERANDS_AND_RESULT(Elu)
OP_SAME_OPERANDS_AND_RESULT(Elu_)
OP_SAME_OPERANDS_AND_RESULT(EmptyLike)
OP_SAME_OPERANDS_AND_RESULT(Erf)
OP_SAME_OPERANDS_AND_RESULT(Erf_)
OP_SAME_OPERANDS_AND_RESULT(Erfinv)
OP_SAME_OPERANDS_AND_RESULT(Erfinv_)
OP_SAME_OPERANDS_AND_RESULT(Exp)
OP_SAME_OPERANDS_AND_RESULT(Exp_)
OP_SAME_OPERANDS_AND_RESULT(Expm1)
OP_SAME_OPERANDS_AND_RESULT(Expm1_)
OP_SAME_OPERANDS_AND_RESULT(Exponential_)
OP_SAME_OPERANDS_AND_RESULT(FakeDequantizeMaxAbs)
OP_SAME_OPERANDS_AND_RESULT(Fill)
OP_SAME_OPERANDS_AND_RESULT(Fill_)
OP_SAME_OPERANDS_AND_RESULT(Fetch)
OP_SAME_OPERANDS_AND_RESULT(Flip)
OP_SAME_OPERANDS_AND_RESULT(Floor)
OP_SAME_OPERANDS_AND_RESULT(Floor_)
OP_SAME_OPERANDS_AND_RESULT(FullLike)
OP_SAME_OPERANDS_AND_RESULT(GetTensorFromSelectedRows)
OP_SAME_OPERANDS_AND_RESULT(Gelu)
OP_SAME_OPERANDS_AND_RESULT(Gelu_)
OP_SAME_OPERANDS_AND_RESULT(Hardswish)
OP_SAME_OPERANDS_AND_RESULT(Imag)
OP_SAME_OPERANDS_AND_RESULT(Increment)
OP_SAME_OPERANDS_AND_RESULT(Increment_)
OP_SAME_OPERANDS_AND_RESULT(Isfinite)
OP_SAME_OPERANDS_AND_RESULT(IsfiniteSr)
OP_SAME_OPERANDS_AND_RESULT(Isinf)
OP_SAME_OPERANDS_AND_RESULT(IsinfSr)
OP_SAME_OPERANDS_AND_RESULT(Isnan)
OP_SAME_OPERANDS_AND_RESULT(IsnanSr)
OP_SAME_OPERANDS_AND_RESULT(I0)
OP_SAME_OPERANDS_AND_RESULT(I0_)
OP_SAME_OPERANDS_AND_RESULT(I0e)
OP_SAME_OPERANDS_AND_RESULT(I1)
OP_SAME_OPERANDS_AND_RESULT(I1e)
OP_SAME_OPERANDS_AND_RESULT(LabelSmooth)
OP_SAME_OPERANDS_AND_RESULT(Lgamma)
OP_SAME_OPERANDS_AND_RESULT(Lgamma_)
OP_SAME_OPERANDS_AND_RESULT(Log1p)
OP_SAME_OPERANDS_AND_RESULT(Log1p_)
OP_SAME_OPERANDS_AND_RESULT(Log)
OP_SAME_OPERANDS_AND_RESULT(Log_)
OP_SAME_OPERANDS_AND_RESULT(Log10)
OP_SAME_OPERANDS_AND_RESULT(Log10_)
OP_SAME_OPERANDS_AND_RESULT(Log2)
OP_SAME_OPERANDS_AND_RESULT(Log2_)
OP_SAME_OPERANDS_AND_RESULT(LogicalNot)
OP_SAME_OPERANDS_AND_RESULT(LogicalNot_)
OP_SAME_OPERANDS_AND_RESULT(Logit)
OP_SAME_OPERANDS_AND_RESULT(Logit_)
OP_SAME_OPERANDS_AND_RESULT(Logsigmoid)
OP_SAME_OPERANDS_AND_RESULT(Logsigmoid_)
OP_SAME_OPERANDS_AND_RESULT(LogSoftmax)
OP_SAME_OPERANDS_AND_RESULT(Memcpy)
OP_SAME_OPERANDS_AND_RESULT(Mish)
OP_SAME_OPERANDS_AND_RESULT(NumberCount)
OP_SAME_OPERANDS_AND_RESULT(Pow)
OP_SAME_OPERANDS_AND_RESULT(Poisson)
OP_SAME_OPERANDS_AND_RESULT(Pow_)
OP_SAME_OPERANDS_AND_RESULT(Prelu)
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
OP_SAME_OPERANDS_AND_RESULT(RowConv)
OP_SAME_OPERANDS_AND_RESULT(Rsqrt)
OP_SAME_OPERANDS_AND_RESULT(Rsqrt_)
OP_SAME_OPERANDS_AND_RESULT(ScaleSr)
OP_SAME_OPERANDS_AND_RESULT(ScaleSr_)
OP_SAME_OPERANDS_AND_RESULT(Scale_)
OP_SAME_OPERANDS_AND_RESULT(ScatterNdAdd)
OP_SAME_OPERANDS_AND_RESULT(Scatter)
OP_SAME_OPERANDS_AND_RESULT(Scatter_)
OP_SAME_OPERANDS_AND_RESULT(Select)
OP_SAME_OPERANDS_AND_RESULT(Selu)
OP_SAME_OPERANDS_AND_RESULT(ShadowFeed)
OP_SAME_OPERANDS_AND_RESULT(ShareData_)
OP_SAME_OPERANDS_AND_RESULT(Sign)
OP_SAME_OPERANDS_AND_RESULT(Sin)
OP_SAME_OPERANDS_AND_RESULT(Sin_)
OP_SAME_OPERANDS_AND_RESULT(Sinh)
OP_SAME_OPERANDS_AND_RESULT(Sinh_)
OP_SAME_OPERANDS_AND_RESULT(Softmax)
OP_SAME_OPERANDS_AND_RESULT(Softmax_)
OP_SAME_OPERANDS_AND_RESULT(Softplus)
OP_SAME_OPERANDS_AND_RESULT(SoftRelu)
OP_SAME_OPERANDS_AND_RESULT(Softshrink)
OP_SAME_OPERANDS_AND_RESULT(Softsign)
OP_SAME_OPERANDS_AND_RESULT(Stanh)
OP_SAME_OPERANDS_AND_RESULT(Swish)
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
OP_SAME_OPERANDS_AND_RESULT(LeakyRelu)
OP_SAME_OPERANDS_AND_RESULT(LeakyRelu_)
OP_SAME_OPERANDS_AND_RESULT(ThresholdedRelu)
OP_SAME_OPERANDS_AND_RESULT(ThresholdedRelu_)
OP_SAME_OPERANDS_AND_RESULT(SquareSr)
OP_SAME_OPERANDS_AND_RESULT(SquareSr_)
OP_SAME_OPERANDS_AND_RESULT(Square)
OP_SAME_OPERANDS_AND_RESULT(Square_)
OP_SAME_OPERANDS_AND_RESULT(Polygamma)
OP_SAME_OPERANDS_AND_RESULT(Polygamma_)
OP_SAME_OPERANDS_AND_RESULT(EnableCheckModelNanInf)
OP_SAME_OPERANDS_AND_RESULT(ViewShape)
OP_SAME_OPERANDS_AND_RESULT(Silu)
OP_SAME_OPERANDS_AND_RESULT(ViewDtype)
OP_SAME_OPERANDS_AND_RESULT(Sqrt)
OP_SAME_OPERANDS_AND_RESULT(Sqrt_)
OP_SAME_OPERANDS_AND_RESULT(SqrtSr)
OP_SAME_OPERANDS_AND_RESULT(SqrtSr_)
OP_SAME_OPERANDS_AND_RESULT(FusedSoftmaxMaskUpperTriangle)
OP_SAME_OPERANDS_AND_RESULT(Gammaln)
OP_SAME_OPERANDS_AND_RESULT(Gammaln_)
OP_SAME_OPERANDS_AND_RESULT(GaussianInplace)
OP_SAME_OPERANDS_AND_RESULT(GaussianInplace_)
OP_SAME_OPERANDS_AND_RESULT(Hardshrink)
OP_SAME_OPERANDS_AND_RESULT(Hardsigmoid)
OP_SAME_OPERANDS_AND_RESULT(MergeSelectedRows)
OP_SAME_OPERANDS_AND_RESULT(NpuIdentity)
OP_SAME_OPERANDS_AND_RESULT(Renorm)
OP_SAME_OPERANDS_AND_RESULT(Renorm_)
OP_SAME_OPERANDS_AND_RESULT(TanhShrink)
OP_SAME_OPERANDS_AND_RESULT(YoloBoxHead)
OP_SAME_OPERANDS_AND_RESULT(StandardGamma)

bool ScaleOpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  pir::Value operand_source = op->operand_source(0);
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(operand_source);
  std::vector<symbol::DimExpr> shape(operand_shape_or_data.shape());

  const auto &SetOutputWithOnlyShape = [&]() {
    infer_context->SetShapeOrDataForValue(
        op->result(0), symbol::TensorShapeOrDataDimExprs(shape));
  };

  const auto &SetOutputWithShapeAndData =
      [&](const std::vector<symbol::DimExpr> &data) {
        infer_context->SetShapeOrDataForValue(
            op->result(0), symbol::TensorShapeOrDataDimExprs(shape, data));
      };

  const auto &GetOptionalAttributeData =
      [&](const std::string &attr_name) -> std::optional<symbol::DimExpr> {
    const auto &float_data =
        op->attribute(attr_name).dyn_cast<pir::FloatAttribute>().data();
    const int64_t &int_data = static_cast<int64_t>(float_data);
    if (float_data - int_data > 1e-6 || float_data - int_data < -1e-6) {
      return std::nullopt;
    }
    return symbol::DimExpr{int_data};
  };

  const auto &GetOptionalScaleData = [&]() -> std::optional<symbol::DimExpr> {
    if (op->num_operands() == 2) {
      const auto &scale_shape_or_data =
          infer_context->GetShapeOrDataForValue(op->operand_source(1));
      if (scale_shape_or_data.data())
        return scale_shape_or_data.data()->at(0);
      else
        return std::nullopt;
    }
    return GetOptionalAttributeData("scale");
  };

  if (operand_shape_or_data.data()) {
    const std::optional<symbol::DimExpr> &opt_scale = GetOptionalScaleData();
    const std::optional<symbol::DimExpr> &opt_bias =
        GetOptionalAttributeData("bias");
    if (opt_scale && opt_bias) {
      std::vector<symbol::DimExpr> data;
      for (auto &val : *(operand_shape_or_data.data())) {
        data.push_back(val * (opt_scale.value()) + (opt_bias.value()));
      }
      SetOutputWithShapeAndData(data);
    } else {
      SetOutputWithOnlyShape();
    }
  } else {
    SetOutputWithOnlyShape();
  }

  return true;
}

bool ArgsortOpInferSymbolicShape(
    pir::Operation *op, pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  infer_context->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  infer_context->SetShapeOrDataForValue(op->result(1), operand_shape_or_data);
  return true;
}

bool CeilOpInferSymbolicShape(pir::Operation *op,
                              pir::InferSymbolicShapeContext *infer_context) {
  const symbol::ShapeOrDataDimExprs &operand_shape_or_data =
      infer_context->GetShapeOrDataForValue(op->operand_source(0));
  infer_context->SetShapeOrDataForValue(op->result(0), operand_shape_or_data);
  return true;
}

bool Ceil_OpInferSymbolicShape(pir::Operation *op,
                               pir::InferSymbolicShapeContext *infer_context) {
  return CeilOpInferSymbolicShape(op, infer_context);
}

}  // namespace paddle::dialect

namespace cinn::dialect {}  // namespace cinn::dialect

#undef OP_SAME_OPERANDS_AND_RESULT
