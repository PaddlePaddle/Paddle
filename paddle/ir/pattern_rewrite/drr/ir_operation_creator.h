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

#include "build/paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/ir/pattern_rewrite/drr/api/drr_pattern_context.h"
#include "paddle/ir/pattern_rewrite/drr/match_context_impl.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace ir {
namespace drr {

std::vector<Value> GetIrValuesByDrrTensors(
    const std::vector<const Tensor*>& tensors,
    const MatchContextImpl& res_match_ctx) {
  std::vector<Value> ir_values;
  ir_values.reserve(tensors.size());
  for (const auto* tensor : tensors) {
    ir_values.push_back(GetIrValueByDrrTensor(*tensor, res_match_ctx));
  }
}

Value GetIrValueByDrrTensor(const Tensor& tensor,
                            const MatchContextImpl& res_match_ctx) {
  return res_match_ctx.GetIrTensor(tensor.id()).ir_value();
}

Operation* CreateOperation(const OpCall& op_call,
                           ir::PatternRewriter& rewriter,  // NOLINT
                           MatchContextImpl* res_match_ctx) {
  if (op_call.name() == "pd.reshape") {
    const auto& inputs = op_call.inputs();
    std::vector<Value> ir_values =
        GetIrValuesByDrrTensors(inputs, res_match_ctx);
    // TODO(zyfncg): support attr in build op.
    Operation* reshape_op = rewriter.Build<paddle::dialect::ReshapeOp>(
        ir_values[0], std::vector<int64_t>{16, 3, 4, 16});
    auto out = reshape_op->result(0);
    res_match_ctx->BindIrTensor(op_call.outputs()[0]->id(),
                                std::make_shared<IrTensor>(out));
    return reshape_op;
  }
  LOG(ERROR) << "Unknown op " << op_call.name();
  return nullptr;
}

}  // namespace drr
}  // namespace ir
