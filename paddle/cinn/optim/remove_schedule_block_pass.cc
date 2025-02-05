// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/remove_schedule_block_pass.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
namespace cinn {
namespace optim {
using ir::stmt::BlockRef;
using ir::stmt::StmtRef;

LogicalResult RemoveScheduleBlockPass::Run(ir::stmt::BlockRef block) {
  const auto& MergeStmtVector = [&](std::vector<StmtRef>& dest,
                                    const std::vector<StmtRef>& source) {
    dest.insert(dest.end(), source.begin(), source.end());
  };

  const std::vector<StmtRef>& stmts = block->stmts();
  std::vector<StmtRef> new_stmts;
  for (const StmtRef& stmt : stmts) {
    if (!stmt.isa<ir::stmt::Schedule>()) {
      new_stmts.push_back(stmt);
      continue;
    }
    const ir::stmt::Schedule schedule_stmt = stmt.as<ir::stmt::Schedule>();
    const std::vector<Expr> iter_values = schedule_stmt->iter_values();
    const BlockRef body = schedule_stmt->body();
    const std::vector<Var> iter_vars = schedule_stmt->iter_vars();
    PADDLE_ENFORCE_EQ(iter_vars.size(),
                      iter_values.size(),
                      ::common::errors::InvalidArgument(
                          "The size of iter vars and iter values is not equal,"
                          "where iter vars:%d but iter values:%d.",
                          iter_vars.size(),
                          iter_values.size()));
    for (int i = 0; i < iter_vars.size(); i++) {
      optim::ReplaceVarWithExpr(body, iter_vars[i], iter_values[i], "");
    }
    MergeStmtVector(new_stmts, body->stmts());
  }
  block->set_stmts(new_stmts);
  return LogicalResult::success();
}

std::unique_ptr<BlockPass> CreateRemoveScheduleBlockPass() {
  return std::make_unique<RemoveScheduleBlockPass>();
}

}  // namespace optim
}  // namespace cinn
