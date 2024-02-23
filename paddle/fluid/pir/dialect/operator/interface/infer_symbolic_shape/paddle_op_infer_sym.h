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

#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace paddle::dialect {

bool DataOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

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

bool ConcatOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool GatherNdOpInferSymbolicShape(
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

bool ProdOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ArangeOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool EmbeddingOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool SparseWeightEmbeddingOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ExpandOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool MatmulOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool MaxOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);

bool TransposeOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool WhereOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);

bool Where_OpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool FeedOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);

bool TopPSamplingOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool ExpandAsOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool SplitOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);

}  // namespace paddle::dialect
