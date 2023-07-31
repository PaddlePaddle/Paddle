// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/ir/pattern_rewrite/drr/api/drr_pass_context.h"

namespace ir {
namespace drr {

class FoldBroadcastToConstantPass : public DrrPass {
 public:
  void operator()(DrrPassContext* ctx) const override {
    // DRR 主体包含三部分： Source Pattern. Constrains， Result Pattern
    // Source patterns：待匹配的子图
    SourcePattern pat = ctx->SourcePattern();
    // Source Pattern 中可匹配的类型包括 Op 和 Tensor
    const auto& fill_constant = pat.Op(
        "fill_constant",
        {{"value", pat.Attr("value_1")}, {"dtype", pat.Attr("dtype_1")}});
    const auto& broadcast_to = pat.Op("broadcast_to");
    // 匹配fill_constant+broadcast_to，同时对输出张量标记为ret，方便后面加约束
    pat.Tensor("ret") = broadcast_to(fill_constant());
    // Constrains：本Pass无额外的约束规则
    // Result patterns：要替换为的子图
    ResultPattern res = pat.ResultPattern();
    // 使用 folded_fill_constant 替换
    // broadcast_to(fill_constant())，注意shape属性已更新 所有 ret
    // 参数均在Source Pattern中使用，对 ret 的赋值等同于对 ret 的 producer
    // op的删除和重连接
    const auto& folded_fill_constant =
        res.Op("fill_constant",
               {{"shape", res.Tensor("ret").shape()},
                {"value", res.Attr("value_1")},
                {"dtype", res.Attr("dtype_1")}});
    res.Tensor("ret") = folded_fill_constant();
  }
};

}  // namespace drr
}  // namespace ir
