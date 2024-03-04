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

#pragma once

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace paddle::dialect {

bool ArgmaxOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool ArgminOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool AsComplexOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool AsRealOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool CummaxOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool CumminOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool CumprodOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Cumprod_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool CumsumOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Cumsum_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool ReshapeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Reshape_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

}  // namespace paddle::dialect
