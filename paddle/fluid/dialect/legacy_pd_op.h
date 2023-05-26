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
REIGSTER_EMPTY_OP(feed, FeedOp);               // To be customized: feed
REIGSTER_EMPTY_OP(batch_norm_, BatchNormOp_);  // To be customized: batch_norm_
REIGSTER_EMPTY_OP(fetch_v2, FetchV2Op);        // To be customized: fetch_v2

}  // namespace dialect
}  // namespace paddle
