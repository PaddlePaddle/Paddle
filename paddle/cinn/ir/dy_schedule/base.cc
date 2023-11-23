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

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/ir/dy_schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace ir {

void DyScheduleImpl::MergeExprs() { CINN_NOT_IMPLEMENTED; }

bool DyScheduleImpl::HasBlock(const std::string& block_name) const {
  auto exprs = module_expr_.GetExprs();
  for (auto& it_expr : exprs) {
    ir::FindBlocksVisitor visitor(block_name);
    auto find_blocks = visitor(&it_expr);
    if (!find_blocks.empty()) {
      CHECK_EQ(find_blocks.size(), 1U)
          << "There should not be more than 1 block with identical name!";
      return true;
    }
  }
  return false;
}

std::vector<Expr> DyScheduleImpl::GetLoops(const Expr& block) const {
  CINN_NOT_IMPLEMENTED;
}

std::vector<Expr> DyScheduleImpl::GetLoops(
    const std::string& block_name) const {
  CINN_NOT_IMPLEMENTED;
}

std::vector<Expr> DyScheduleImpl::GetAllBlocks() const { CINN_NOT_IMPLEMENTED; }

std::vector<Expr> DyScheduleImpl::GetChildBlocks(const Expr& expr) const {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::GetBlock(const std::string& block_name) const {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::GetRootBlock(const Expr& expr) const {
  auto exprs = this->GetModule().GetExprs();
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

DeviceAPI DyScheduleImpl::GetDeviceAPI() const { CINN_NOT_IMPLEMENTED; }

void DyScheduleImpl::Annotate(const Expr& block,
                              const std::string& key,
                              const attr_t& value) {
  CINN_NOT_IMPLEMENTED;
}

void DyScheduleImpl::Unannotate(Expr& block,
                                const std::string& key) {  // NOLINT
  CINN_NOT_IMPLEMENTED;
}

void DyScheduleImpl::CopyTransformAndLoopInfo(const Expr& block,
                                              const Expr& block_target) {
  CINN_NOT_IMPLEMENTED;
}

void DyScheduleImpl::CopyTransformAndLoopInfo(
    const std::string& block_name, const std::string& block_target_name) {
  CINN_NOT_IMPLEMENTED;
}

Expr DyScheduleImpl::SampleCategorical(
    utils::LinearRandomEngine::StateType* rand_seed,
    const std::vector<int>& candidates,
    const std::vector<float>& probs) {
  CINN_NOT_IMPLEMENTED;
}

std::vector<Expr> DyScheduleImpl::SamplePerfectTile(
    utils::LinearRandomEngine::StateType* rand_seed,
    const Expr& loop,
    int n,
    int max_innermost_factor) {
  CINN_NOT_IMPLEMENTED;
}

}  // namespace ir
}  // namespace cinn
