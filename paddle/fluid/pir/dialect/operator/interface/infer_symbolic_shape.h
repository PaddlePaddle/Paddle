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

#pragma once

#include "paddle/pir/core/op_base.h"
#include "paddle/pir/dialect/shape/utils/shape_analysis.h"

// Type inference is currently modelled executionally for operation creation
// using the `InferMetaInterface`. While `InferSymbolicShapeInterface` is used
// to implement the shape and element type inference. The return type can often
// be deduced from the deduced return shape and elemental type (queryable from
// `InferSymbolicShapeInterface`) and so type inference for tensor types can be
// implemented with `InferSymbolicShapeInterface`.

namespace paddle::dialect {

class InferSymbolicShapeInterface
    : public pir::OpInterfaceBase<InferSymbolicShapeInterface> {
 public:
  /// Defined these methods with the interface.
  struct Concept {
    explicit Concept(bool (*infer_symbolic_shapes)(
        pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis))
        : infer_symbolic_shapes(infer_symbolic_shapes) {}
    bool (*infer_symbolic_shapes)(
        pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
  };

  template <class ConcreteOp>
  struct Model : public Concept {
    static inline bool InferSymbolicShape(
        pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
      return op->dyn_cast<ConcreteOp>().InferSymbolicShape(shape_analysis);
    }

    Model() : Concept(InferSymbolicShape) {}
  };

  /// Constructor
  InferSymbolicShapeInterface(pir::Operation *op, Concept *impl)
      : pir::OpInterfaceBase<InferSymbolicShapeInterface>(op), impl_(impl) {}

  bool InferSymbolicShape(pir::ShapeConstraintIRAnalysis *shape_analysis);

 private:
  Concept *impl_;
};

}  // namespace paddle::dialect

namespace paddle::dialect {

bool AbsOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Abs_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool DataOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool AddOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Add_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool CastOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Cast_OpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ExpOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Exp_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool SubtractOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Subtract_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ShapeOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);
bool ShapeSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool StackOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);

bool SumOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool FullIntArrayOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);

bool FullOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool MultiplyOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool MultiplySrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Multiply_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool MultiplySr_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ConcatOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool GatherNdOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool PowOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Pow_OpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool RsqrtOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Rsqrt_OpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ScaleOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Scale_OpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool ScaleSrOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool ScaleSr_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool SqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Squeeze_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool UnsqueezeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Unsqueeze_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool TileOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool TransposeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Transpose_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool DivideOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Divide_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ElementwisePowOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool FullWithTensorOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ReluOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Relu_OpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);
}  // namespace paddle::dialect

namespace cinn::dialect {

bool SliceOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);
bool ScaleOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);
}  // namespace cinn::dialect

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::InferSymbolicShapeInterface)
