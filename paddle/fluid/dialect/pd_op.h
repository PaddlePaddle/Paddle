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

#include "paddle/ir/op_base.h"

namespace paddle {
namespace dialect {

#define OPNAME(op_name) "pd." #op_name

#define REIGSTER_EMPTY_OP(op_name, className)             \
  class className : public ir::Op<className> {            \
   public:                                                \
    static const char *name() { return OPNAME(op_name); } \
    static const char **attributes_name;                  \
    static constexpr uint32_t attributes_num = 0;         \
  };                                                      \
  const char **className::attributes_name = nullptr;

REIGSTER_EMPTY_OP(conv2d, Conv2DOp);
REIGSTER_EMPTY_OP(feed, FeedOp);
REIGSTER_EMPTY_OP(batch_norm, BatchNormOp);
REIGSTER_EMPTY_OP(batch_norm_, BatchNormOp_);
REIGSTER_EMPTY_OP(relu, ReluOp);
REIGSTER_EMPTY_OP(elementwise_add, ElementwiseAddOp);
REIGSTER_EMPTY_OP(pool2d, Pool2DOp);
REIGSTER_EMPTY_OP(flatten_contiguous_range, FlattenContiguousRangeOp);
REIGSTER_EMPTY_OP(matmul_v2, MatmulV2Op);
REIGSTER_EMPTY_OP(reshape2, Reshape2Op);
REIGSTER_EMPTY_OP(softmax_with_cross_entropy, SoftmaxWithCrossEntropyOp);
REIGSTER_EMPTY_OP(reduce_mean, ReduceMeanOp);
REIGSTER_EMPTY_OP(top_k_v2, TopKV2Op);
REIGSTER_EMPTY_OP(scale, ScaleOp);
REIGSTER_EMPTY_OP(accuracy, AccuracyOp);
REIGSTER_EMPTY_OP(fill_constant, FillConstantOp);
REIGSTER_EMPTY_OP(reduce_mean_grad, ReduceMeanGradOp);
REIGSTER_EMPTY_OP(softmax_with_cross_entropy_grad,
                  SoftmaxWithCrossEntropyGradOp);
REIGSTER_EMPTY_OP(elementwise_add_grad, ElementwiseAddGradOp);
REIGSTER_EMPTY_OP(matmul_v2_grad, MatmulV2GradOp);
REIGSTER_EMPTY_OP(flatten_contiguous_range_grad, FlattenContiguousRangeGradOp);
REIGSTER_EMPTY_OP(pool2d_grad, Pool2DGradOp);
REIGSTER_EMPTY_OP(relu_grad, ReluGradOp);
REIGSTER_EMPTY_OP(batch_norm_grad, BatchNormGradOp);
REIGSTER_EMPTY_OP(conv2d_grad, Conv2DGradOp);
REIGSTER_EMPTY_OP(sum, SumOp);
REIGSTER_EMPTY_OP(fetch_v2, FetchV2Op);
REIGSTER_EMPTY_OP(merged_momentum_, MergedMomentumOp_);

}  // namespace dialect
}  // namespace paddle
