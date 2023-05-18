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

#define REIGSTER_EMPTY_OP(op_name)                        \
  class op_name##Op : public ir::Op<op_name##Op> {        \
   public:                                                \
    static const char *name() { return OPNAME(op_name); } \
    static const char **attributes_name_;                 \
    static uint32_t attributes_num() { return 0; }        \
  };                                                      \
  const char **op_name##Op::attributes_name_ = nullptr;

REIGSTER_EMPTY_OP(conv2d);
REIGSTER_EMPTY_OP(feed);
REIGSTER_EMPTY_OP(batch_norm);
REIGSTER_EMPTY_OP(batch_norm_);
REIGSTER_EMPTY_OP(relu);
REIGSTER_EMPTY_OP(elementwise_add);
REIGSTER_EMPTY_OP(pool2d);
REIGSTER_EMPTY_OP(flatten_contiguous_range);
REIGSTER_EMPTY_OP(matmul_v2);
REIGSTER_EMPTY_OP(reshape2);
REIGSTER_EMPTY_OP(softmax_with_cross_entropy);
REIGSTER_EMPTY_OP(reduce_mean);
REIGSTER_EMPTY_OP(top_k_v2);
REIGSTER_EMPTY_OP(scale);
REIGSTER_EMPTY_OP(accuracy);
REIGSTER_EMPTY_OP(fill_constant);
REIGSTER_EMPTY_OP(reduce_mean_grad);
REIGSTER_EMPTY_OP(softmax_with_cross_entropy_grad);
REIGSTER_EMPTY_OP(elementwise_add_grad);
REIGSTER_EMPTY_OP(matmul_v2_grad);
REIGSTER_EMPTY_OP(flatten_contiguous_range_grad);
REIGSTER_EMPTY_OP(pool2d_grad);
REIGSTER_EMPTY_OP(relu_grad);
REIGSTER_EMPTY_OP(batch_norm_grad);
REIGSTER_EMPTY_OP(conv2d_grad);
REIGSTER_EMPTY_OP(sum);
REIGSTER_EMPTY_OP(fetch_v2);
REIGSTER_EMPTY_OP(merged_momentum_);

}  // namespace dialect
}  // namespace paddle
