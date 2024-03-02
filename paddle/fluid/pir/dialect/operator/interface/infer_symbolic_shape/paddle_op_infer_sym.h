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

//  Not Impelmented Ops.

bool ArgmaxOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool ArgminOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool AsComplexOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool AsRealOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool AsStridedOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

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

bool DiagEmbedOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool DiagonalOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool DirichletOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool GatherOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

bool KronOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);
bool KthvalueOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool LogcumsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool MaskedSelectOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool PoissonOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool PutAlongAxisOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool PutAlongAxis_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool SearchsortedOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool TakeAlongAxisOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool TopkOpInferSymbolicShape(pir::Operation *op,
                              pir::ShapeConstraintIRAnalysis *shape_analysis);
bool UnbindOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool UniqueConsecutiveOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);

bool EinsumOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);
bool EmptyOpInferSymbolicShape(pir::Operation *op,
                               pir::ShapeConstraintIRAnalysis *shape_analysis);
bool Exponential_OpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool GaussianOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool LinspaceOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool LogspaceOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool LogsumexpOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool MinOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);
bool PadOpInferSymbolicShape(pir::Operation *op,
                             pir::ShapeConstraintIRAnalysis *shape_analysis);
bool RandintOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool RepeatInterleaveOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool SplitWithNumOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool TrilIndicesOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool TriuIndicesOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool UniformOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis);
bool UniqueOpInferSymbolicShape(pir::Operation *op,
                                pir::ShapeConstraintIRAnalysis *shape_analysis);

}  // namespace paddle::dialect
