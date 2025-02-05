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

#pragma once
#include "paddle/cinn/optim/if_fusion_pass.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_compare.h"

namespace cinn {
namespace optim {
using ir::stmt::BlockRef;
using ir::stmt::IfThenElse;
using ir::stmt::StmtRef;

void FuseIfStmtWithSameCondInBlock(BlockRef block) {
  const std::vector<StmtRef>& stmts = block->stmts();
  if (stmts.size() < 2) return;

  const auto& IsIfStmtWithSpecCond = [](const StmtRef& stmt, const Expr& cond) {
    if (!stmt.isa<IfThenElse>()) return false;
    const auto& if_stmt = stmt.as<IfThenElse>();
    return ir::ir_utils::IRCompare(if_stmt->condition(), cond);
  };

  const auto& AppendBlockStmts = [](BlockRef dest, const BlockRef& source) {
    const auto& source_stmts = source->stmts();
    std::vector<StmtRef> new_stmts = dest->stmts();
#ifdef __cpp_lib_containers_range
    new_stmts.append_range(source_stmt);
#else
    new_stmts.insert(
        new_stmts.end(), source_stmts.cbegin(), source_stmts.cend());
#endif
    dest->set_stmts(new_stmts);
  };

  const auto& FuseIfStmt =
      [&AppendBlockStmts](
          const std::vector<IfThenElse>& if_stmts) -> IfThenElse {
    BlockRef new_true_case{std::vector<StmtRef>()};
    BlockRef new_false_case{std::vector<StmtRef>()};
    for (const auto& if_stmt : if_stmts) {
      AppendBlockStmts(new_true_case, if_stmt->true_case());
      AppendBlockStmts(new_false_case, if_stmt->false_case());
    }
    FuseIfStmtWithSameCondInBlock(new_true_case);
    FuseIfStmtWithSameCondInBlock(new_false_case);
    return IfThenElse(if_stmts[0]->condition(), new_true_case, new_false_case);
  };

  struct CandidateIfStmtGroup {
    int start_if_idx;
    int end_if_idx;
    CandidateIfStmtGroup(const int& start, const int& end)
        : start_if_idx(start), end_if_idx(end) {}
  };

  const auto& candidates = [&]() -> std::vector<CandidateIfStmtGroup> {
    std::vector<CandidateIfStmtGroup> group_infos;
    size_t idx = 0;
    while (idx < stmts.size() - 1) {
      if (!stmts[idx].isa<IfThenElse>()) {
        ++idx;
        continue;
      }
      int start_idx = idx;
      const auto& start_if = stmts[start_idx].as<IfThenElse>();
      int stmt_num_in_group = 1;
      for (std::size_t next_idx = start_idx + 1; next_idx < stmts.size();
           ++next_idx) {
        if (IsIfStmtWithSpecCond(stmts[next_idx], start_if->condition())) {
          ++stmt_num_in_group;
        } else {
          break;
        }
      }
      if (stmt_num_in_group > 1) {
        CandidateIfStmtGroup group(start_idx,
                                   start_idx + stmt_num_in_group - 1);
        group_infos.push_back(group);
      }
      idx += stmt_num_in_group;
    }
    return group_infos;
  }();

  if (candidates.empty()) return;

  const auto& new_stmts = [&]() -> std::vector<StmtRef> {
    std::vector<StmtRef> res;
    int remain_stmt_idx = 0;
    for (const auto& group : candidates) {
      for (int i = remain_stmt_idx; i < group.start_if_idx; ++i) {
        res.emplace_back(stmts[i]);
      }
      const auto& if_stmts_to_be_fused = [&]() -> std::vector<IfThenElse> {
        std::vector<IfThenElse> if_stmt_group;
        for (int i = group.start_if_idx; i <= group.end_if_idx; ++i) {
          if_stmt_group.emplace_back(stmts[i].as<IfThenElse>());
        }
        return if_stmt_group;
      }();
      res.emplace_back(FuseIfStmt(if_stmts_to_be_fused));
      remain_stmt_idx = group.end_if_idx + 1;
    }
    for (; remain_stmt_idx < stmts.size(); ++remain_stmt_idx) {
      res.emplace_back(stmts[remain_stmt_idx]);
    }
    return res;
  }();

  block->set_stmts(new_stmts);
}

LogicalResult IfFusionPass::Run(ir::stmt::BlockRef block) {
  FuseIfStmtWithSameCondInBlock(block);
  return LogicalResult::success();
}

std::unique_ptr<BlockPass> CreateIfFusionPass() {
  return std::make_unique<IfFusionPass>();
}
}  // namespace optim
}  // namespace cinn
