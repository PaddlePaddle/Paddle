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

#pragma once

#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/schedule/schedule_base.h"
#include "paddle/cinn/ir/schedule/schedule_desc.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/utils/error.h"
#include "paddle/cinn/utils/random_engine.h"

namespace cinn {
namespace ir {
namespace analyzer {

bool HasBlock(const std::vector<Expr>& exprs, const std::string& block_name) {
  for (auto& it_expr : exprs) {
    FindBlocksVisitor visitor(block_name);
    auto find_blocks = visitor(&it_expr);
    if (!find_blocks.empty()) {
      CHECK_EQ(find_blocks.size(), 1U)
          << "There should not be more than 1 block with identical name!";
      return true;
    }
  }
  return false;
}

std::vector<Expr> GetLoops(const std::vector<Expr>& exprs,
                           const std::string& block_name) {
  Expr block = GetBlock(exprs, block_name);
  std::vector<Expr> result = GetLoops(exprs, block);
  return result;
}

std::vector<Expr> GetLoops(const std::vector<Expr>& exprs, const Expr& block) {
  std::vector<Expr> result;
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  std::string block_name = block.As<ir::ScheduleBlockRealize>()
                               ->schedule_block.As<ir::ScheduleBlock>()
                               ->name;

  for (auto& it_expr : exprs) {
    FindLoopsVisitor visitor(block);
    auto find_loops = visitor(&it_expr);
    if (!find_loops.empty()) {
      if (!result.empty())
        LOG(FATAL) << "Find block with name: \n"
                   << block_name << " appeared in more than one AST!";
      result = find_loops;
    }
  }

  if (result.empty()) {
    result.push_back(AddUnitLoop(exprs, block));
  }
  return result;
}

std::vector<Expr> GetAllBlocks(const std::vector<Expr>& exprs) {
  std::vector<Expr> result;
  for (auto& it_expr : exprs) {
    FindBlocksVisitor visitor;
    auto find_blocks = visitor(&it_expr);
    result.insert(result.end(), find_blocks.begin(), find_blocks.end());
  }
  for (auto& it_expr : exprs) {
    VLOG(3) << "it_expr is : " << it_expr;
  }
  CHECK(!result.empty()) << "Didn't find blocks in expr.";
  return result;
}

std::vector<Expr> GetChildBlocks(const Expr& expr) {
  CHECK(expr.As<ir::ScheduleBlockRealize>() || expr.As<ir::For>());
  FindBlocksVisitor visitor;
  std::vector<Expr> result = visitor(&expr);
  return result;
}

Expr GetBlock(const std::vector<Expr>& exprs, const std::string& block_name) {
  Expr result;
  for (auto& it_expr : exprs) {
    FindBlocksVisitor visitor(block_name);
    auto find_blocks = visitor(&it_expr);
    if (!find_blocks.empty()) {
      CHECK_EQ(find_blocks.size(), 1U)
          << "There should not be more than 1 block with identical name!";
      result = find_blocks[0];
      return result;
    }
  }
  LOG(FATAL) << "Didn't find a block with name " << block_name
             << " in this ModuleExpr!";
}

Expr GetRootBlock(const std::vector<Expr>& exprs, const Expr& expr) {
  for (auto& it_expr : exprs) {
    auto find_expr = ir::ir_utils::CollectIRNodesWithoutTensor(
        it_expr,
        [&](const Expr* x) {
          return x->node_type() == expr.node_type() && *x == expr;
        },
        true);
    if (!find_expr.empty()) {
      CHECK(it_expr.As<ir::Block>());
      CHECK_EQ(it_expr.As<ir::Block>()->stmts.size(), 1U);
      CHECK(it_expr.As<ir::Block>()->stmts[0].As<ir::ScheduleBlockRealize>());
      return it_expr.As<ir::Block>()->stmts[0];
    }
  }
  LOG(FATAL) << "Didn't find expr \n"
             << expr << "in StScheduleImpl:\n"
             << exprs[0];
}

DeviceAPI GetDeviceAPI(const std::vector<Expr>& exprs) {
  auto find_for_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
      exprs.front(), [&](const Expr* x) { return x->As<ir::For>(); }, true);
  CHECK(!find_for_nodes.empty());
  return (*find_for_nodes.begin()).As<ir::For>()->device_api;
}

Expr AddUnitLoop(const std::vector<Expr>& exprs, const Expr& block) {
  CHECK(block.As<ir::ScheduleBlockRealize>());
  CHECK(block.As<ir::ScheduleBlockRealize>()
            ->schedule_block.As<ir::ScheduleBlock>());
  std::string block_name = block.As<ir::ScheduleBlockRealize>()
                               ->schedule_block.As<ir::ScheduleBlock>()
                               ->name;

  FindBlockParent visitor(block_name);
  for (auto expr : exprs) {
    visitor(&expr);
    if (visitor.target_) {
      break;
    }
  }

  CHECK(visitor.target_) << ", block name : " << block_name << "\n" << exprs;
  if (visitor.target_->As<ir::Block>()) {
    for (auto& stmt : visitor.target_->As<ir::Block>()->stmts) {
      if (stmt.As<ir::ScheduleBlockRealize>()) {
        if (stmt.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name == block_name) {
          auto block = ir::Block::Make({GetBlock(exprs, block_name)});
          auto loop = ir::For::Make(ir::Var(cinn::common::UniqName("ix")),
                                    ir::Expr(0),
                                    ir::Expr(1),
                                    ir::ForType::Serial,
                                    ir::DeviceAPI::UNK,
                                    block);
          stmt = loop;
          return loop;
        }
      }
    }
  } else if (visitor.target_->As<ir::For>()) {
    auto block = ir::Block::Make({visitor.target_->As<ir::For>()->body});
    auto loop = ir::For::Make(ir::Var(cinn::common::UniqName("ix")),
                              ir::Expr(0),
                              ir::Expr(1),
                              ir::ForType::Serial,
                              ir::DeviceAPI::UNK,
                              block);
    visitor.target_->As<ir::For>()->body = loop;
    return loop;
  } else if (visitor.target_->As<ir::ScheduleBlock>()) {
    auto block =
        ir::Block::Make({visitor.target_->As<ir::ScheduleBlock>()->body});
    auto loop = ir::For::Make(ir::Var(cinn::common::UniqName("ix")),
                              ir::Expr(0),
                              ir::Expr(1),
                              ir::ForType::Serial,
                              ir::DeviceAPI::UNK,
                              block);
    visitor.target_->As<ir::ScheduleBlock>()->body = loop;
    return loop;
  } else {
    LOG(FATAL) << "Can't find block's parent!";
  }
  LOG(FATAL) << "Shouldn't reach code here in AddUnitLoop";
  return Expr{nullptr};
}

}  // namespace analyzer
}  // namespace ir
}  // namespace cinn
