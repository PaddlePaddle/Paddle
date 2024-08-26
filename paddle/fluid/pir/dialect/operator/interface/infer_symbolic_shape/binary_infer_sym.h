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

OP_DECLARE_INFER_SYMBOLIC_SHAPE(Allclose)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ApplyPerChannelScale)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Atan2)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BceLoss)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BceLoss_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BoxClip)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Binomial)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Binomial_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Bincount)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Bmm)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(CholeskySolve)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CtcAlign)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Conv2d)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Conv2dTranspose)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Conv3d)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(ConvTranspose)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Cross)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Correlation)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(DepthwiseConv)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Dot)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Dropout)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Embedding)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(EqualAll)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SparseWeightEmbedding)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ExpandAs)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FillDiagonalTensor)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FillDiagonalTensor_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedSoftmaxMask)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(GridSample)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Gather)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GatherNd)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(GatherTree)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(HuberLoss)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Histogram)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Isclose)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexAdd)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexAdd_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexPut)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexPut_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexSelect)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AccuracyCheck)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexSample)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(IndexSelectStrided)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(KldivLoss)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Kron)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Lstsq)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LuUnpack)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MatrixRankTol)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MaskedSelect)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Matmul)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MatrixNms)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MarginCrossEntropy)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MatmulWithFlatten)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Mv)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(PriorBox)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(PullBoxSparse)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(PullGpuPsSparse)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(PullSparseV2)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(RepeatInterleaveWithTensorIndex)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ReduceAs)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Searchsorted)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SegmentPool)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SequenceMask)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(ShuffleBatch)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Solve)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Stft)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Swiglu)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TakeAlongAxis)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TopPSampling)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TdmChild)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(TriangularSolve)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Unpool3d)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Unpool)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(WeightDequantize)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(YoloBox)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(YoloBoxHead)

}  // namespace paddle::dialect

namespace cinn::dialect {
using paddle::dialect::IscloseOpInferSymbolicShape;
}
