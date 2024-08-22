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

OP_DECLARE_INFER_SYMBOLIC_SHAPE(Accuracy)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Addmm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Addmm_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(AddN)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Auc)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(AssignPos)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(BroadcastTensors)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BatchFc)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BatchNorm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BatchNorm_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BicubicInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Bilinear)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(BilinearInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CheckFiniteAndUnscale)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CheckFiniteAndUnscale_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CrfDecoding)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Concat)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CrossEntropyWithSoftmax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(CrossEntropyWithSoftmax_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(CudnnLstm)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(DeformableConv)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(DetectionMap)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FakeQuantizeDequantizeMovingAverageAbsMax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FakeQuantizeDequantizeMovingAverageAbsMax_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FakeQuantizeMovingAverageAbsMax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FakeQuantizeMovingAverageAbsMax_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(CoalesceTensor)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(CoalesceTensor_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(EditDistance)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FakeQuantizeRangeAbsMax)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FakeQuantizeRangeAbsMax_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FullWithTensor)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FlashAttn)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedFeedforward)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedAttention)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(FlashAttnQkvpacked)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(FlashAttnUnpadded)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedBatchNormAct)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedBatchNormAct_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedBnAddActivation)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedBnAddActivation_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FusedMultiTransformer)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(GenerateProposals)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(GraphKhopSampler)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(GraphReindex)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(GraphSampleNeighbors)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Gru)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(GruUnit)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(GroupNorm)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(InstanceNorm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Lerp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Lerp_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LayerNorm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Linspace)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(LinearInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Logspace)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Lstm)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MaskedMultiheadAttention_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MergedAdam)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MergedAdam_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MergedMomentum)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(MergedMomentum_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MulticlassNms3)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MemoryEfficientAttention)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Meshgrid)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Moe)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MovingAverageAbsMaxScale)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(MovingAverageAbsMaxScale_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(NearestInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(QuantizeLinear)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(QuantizeLinear_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(DequantizeLinear)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(DequantizeLinear_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Nce)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(PsroiPool)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(PyramidHash)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(QuantizeLinear)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(QuantizeLinear_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(RankAttention)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(RmsNorm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(RoiPool)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Rnn)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(RoiAlign)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SpectralNorm)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(SequenceConv)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(SpectralNorm)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SparseAttention)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Stack)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SigmoidCrossEntropyWithLogits)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(SigmoidCrossEntropyWithLogits_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(SyncBatchNorm)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(SyncBatchNorm_)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(SaveCombine)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TdmSampler)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(TrilinearInterp)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(HsigmoidLoss)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(ViterbiDecode)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(Warpctc)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Warprnnt)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(WeightOnlyLinear)
// OP_DECLARE_INFER_SYMBOLIC_SHAPE(WeightedSampleNeighbors)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Where)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Where_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(YoloLoss)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(FakeChannelWiseDequantizeMaxAbs)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(UpdateLossScaling_)
OP_DECLARE_INFER_SYMBOLIC_SHAPE(YoloBoxPost)

}  // namespace paddle::dialect
