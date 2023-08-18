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

#include <vector>

#include "paddle/fluid/ir/drr/api/drr_pattern_context.h"
#include "paddle/fluid/ir/drr/match_context_impl.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace ir {
namespace drr {

Operation* CreateOperation(const OpCall& op_call,
                           const MatchContextImpl& src_match_ctx,
                           ir::PatternRewriter& rewriter,  // NOLINT
                           MatchContextImpl* res_match_ctx);

// template <typename Op>
// class CreateOperation {
//  public:
//   Operation* operator()(const OpCall& op_call,
//                         ir::PatternRewriter& rewriter,  // NOLINT
//                         MatchContextImpl* res_match_ctx) {
//     IR_THROW("Not implemented");
//   }
// };

// template <>
// class CreateOperation<paddle::dialect::ReshapeOp> {
//  public:
//   Operation* operator()(const OpCall& op_call,
//                         ir::PatternRewriter& rewriter,  // NOLINT
//                         MatchContextImpl* res_match_ctx) {
//     const auto& inputs = op_call.inputs();
//     std::vector<Value> ir_values =
//         GetIrValuesByDrrTensors(inputs, *res_match_ctx);
//     // TODO(zyfncg): support attr in build op.
//     Operation* reshape_op = rewriter.Build<paddle::dialect::ReshapeOp>(
//         ir_values[0].dyn_cast<ir::OpResult>(),
//         std::vector<int64_t>{16, 3, 4, 16});
//     auto out = reshape_op->result(0);
//     res_match_ctx->BindIrValue(op_call.outputs()[0]->name(),
//                                std::make_shared<IrValue>(out));
//     return reshape_op;
//   }
// };

}  // namespace drr
}  // namespace ir
