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

#define REIGSTER_EMPTY_OP(op_name, className)                   \
  class className : public ir::Op<className> {                  \
   public:                                                      \
    static const char *name() { return OPNAME(op_name); }       \
    static constexpr const char **attributes_name = nullptr;    \
    static constexpr uint32_t attributes_num = 0;               \
    static void verify(const std::vector<ir::OpResult> &inputs, \
                       const std::vector<ir::Type> &outputs,    \
                       const ir::AttributeMap &attributes) {    \
      LOG(WARNING) << "This is a fake verify";                  \
    }                                                           \
  };

// TODO(zhangbo): As operators are supplemented and defined, they are gradually
// removed.
REIGSTER_EMPTY_OP(conv2d, Conv2DOp);           // To be customized: conv2d
REIGSTER_EMPTY_OP(feed, FeedOp);               // To be customized: feed
REIGSTER_EMPTY_OP(batch_norm, BatchNormOp);    // To be customized: batch_norm
REIGSTER_EMPTY_OP(batch_norm_, BatchNormOp_);  // To be customized: batch_norm_
REIGSTER_EMPTY_OP(elementwise_add,
                  ElementwiseAddOp);  // To be customized: add (elementwise_add)
REIGSTER_EMPTY_OP(pool2d, Pool2DOp);  // To be customized: pool2d
REIGSTER_EMPTY_OP(
    flatten_contiguous_range,
    FlattenContiguousRangeOp);  //  flatten (flatten_contiguous_range)
REIGSTER_EMPTY_OP(matmul_v2,
                  MatmulV2Op);  // To be customized: matmul (matmul_v2)
REIGSTER_EMPTY_OP(reshape2, Reshape2Op);  // To be customized: reshape
REIGSTER_EMPTY_OP(softmax_with_cross_entropy,
                  SoftmaxWithCrossEntropyOp);  // cross_entropy_with_softmax
                                               // (softmax_with_cross_entropy)
REIGSTER_EMPTY_OP(reduce_mean,
                  ReduceMeanOp);        // To be customized: mean (reduce_mean)
REIGSTER_EMPTY_OP(top_k_v2, TopKV2Op);  //  topk (top_k_v2)
REIGSTER_EMPTY_OP(fill_constant,
                  FillConstantOp);  // To be customized: full (fill_constant)
REIGSTER_EMPTY_OP(reduce_mean_grad,
                  ReduceMeanGradOp);  // To be customized: reduce_mean_grad
REIGSTER_EMPTY_OP(
    softmax_with_cross_entropy_grad,
    SoftmaxWithCrossEntropyGradOp);  // cross_entropy_with_softmax_grad
                                     // (softmax_with_cross_entropy_grad)
REIGSTER_EMPTY_OP(
    elementwise_add_grad,
    ElementwiseAddGradOp);  // To be customized: add_grad (elementwise_add_grad)
REIGSTER_EMPTY_OP(
    matmul_v2_grad,
    MatmulV2GradOp);  // To be customized: matmul_grad (matmul_v2_grad)
REIGSTER_EMPTY_OP(
    flatten_contiguous_range_grad,
    FlattenContiguousRangeGradOp);  //   flatten_grad
                                    //   (flatten_contiguous_range_grad)
REIGSTER_EMPTY_OP(pool2d_grad, Pool2DGradOp);  // To be customized: pool2d_grad
REIGSTER_EMPTY_OP(batch_norm_grad,
                  BatchNormGradOp);  // To be customized: batch_norm_grad
REIGSTER_EMPTY_OP(conv2d_grad, Conv2DGradOp);  // To be customized: conv2d_grad
REIGSTER_EMPTY_OP(sum, SumOp);           // To be customized: sum(reduce_sum)
REIGSTER_EMPTY_OP(fetch_v2, FetchV2Op);  // To be customized: fetch_v2

}  // namespace dialect
}  // namespace paddle
