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

#include "paddle/fluid/pir/transforms/build_cinn_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;
using CompatibleInfo = cinn::hlir::framework::pir::CompatibleInfo;

void VerifyOperationOrder(const pir::Block& block);

std::string FormatDuration(std::chrono::milliseconds ms);

class BuildCinnPass : public pir::Pass {
 public:
  BuildCinnPass() : pir::Pass("build_cinn_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      for (auto& block : op->region(i)) {
        ProcessBlock(&block);
        VerifyOperationOrder(block);
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0 && !op->isa<cinn::dialect::GroupOp>() &&
           !op->isa<cinn::dialect::FusionOp>();
  }

 private:
  void ProcessBlock(pir::Block* block) {
    auto start_t = std::chrono::high_resolution_clock::now();
    auto num_ops = block->size();
    std::vector<GroupOpsVec> groups =
        ::pir::DetectSubGraphs(block, CompatibleInfo::IsSupportForCinn);
    AddStatistics(groups.size());
    for (auto& group_ops : groups) {
      if (group_ops.size() == 1 && group_ops[0]->name() == "pd_op.full") {
        continue;
      }
      VLOG(4) << "current group_ops.size(): " << group_ops.size();
      ::pir::ReplaceWithGroupOp(block, group_ops);
    }
    auto end_t = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t);
    LOG(INFO) << "Time of building group ops (size=" << num_ops
              << "): " << FormatDuration(duration);
  }
};

static void VLOG_LINES(const std::string& str) {
  const auto& lines = cinn::utils::Split(str, "\n");
  for (const auto& line : lines) {
    VLOG(4) << line;
  }
}

static std::string OpsDebugStr(std::vector<pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(*op);
    ss << "{" << op->id() << "}\n";
  }
  return ss.str();
}

void VerifyOperationOrder(const pir::Block& block) {
  std::vector<pir::Operation*> block_ops;
  for (auto& op : block) {
    block_ops.push_back(&op);
  }
  VLOG(4) << "VerifyOperationOrder: Block ops is: " << block_ops.size();
  VLOG_LINES(OpsDebugStr(block_ops));
  auto order_info =
      [&]() -> std::unordered_map<const pir::Operation*, int64_t> {
    std::unordered_map<const pir::Operation*, int64_t> map;
    // initialize the position index with block size by default.
    const int64_t block_size = block.size();
    for (auto& op : block) map[&op] = block_size;
    return map;
  }();
  const auto& CheckOpOrder = [&](const pir::Operation* op) -> void {
    const pir::Operation* current_op = op;
    for (auto& value : op->operands_source()) {
      if (!value || !value.defining_op()) continue;
      pir::Operation* defining_op = value.defining_op();
      if (order_info.count(defining_op) == 0) continue;
      if (op->GetParentOp() &&
          op->GetParentOp()->isa<cinn::dialect::GroupOp>()) {
        current_op = op->GetParentOp();
      }
      std::stringstream error_msg;
      error_msg << "The order of operations is not correct! "
                << "Received defining_op(" << defining_op->id() << " "
                << order_info.at(defining_op) << ") is behind current_op("
                << current_op->id() << " " << order_info.at(current_op) << ")";
      PADDLE_ENFORCE_LT(order_info.at(defining_op),
                        order_info.at(current_op),
                        common::errors::Fatal(error_msg.str()));
    }
  };

  const auto& CheckGroupOpOrder = [&](pir::Operation* op) -> void {
    auto group_op = op->dyn_cast<cinn::dialect::GroupOp>();
    for (auto& inner_op : *group_op.block()) {
      CheckOpOrder(&inner_op);
    }
  };

  int64_t index = 0;
  for (auto& op : block) {
    order_info[&op] = index++;
    if (op.isa<cinn::dialect::GroupOp>()) {
      CheckGroupOpOrder(&op);
    } else {
      CheckOpOrder(&op);
    }
  }
}

std::string FormatDuration(std::chrono::milliseconds ms) {
  auto minutes = std::chrono::duration_cast<std::chrono::minutes>(ms);
  ms -= std::chrono::duration_cast<std::chrono::milliseconds>(minutes);
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(ms);
  ms -= std::chrono::duration_cast<std::chrono::milliseconds>(seconds);
  std::stringstream ss;
  ss << minutes.count() << " min " << seconds.count() << " s " << ms.count()
     << " ms";
  return ss.str();
}

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateBuildCinnPass() {
  return std::make_unique<BuildCinnPass>();
}

}  // namespace pir

REGISTER_IR_PASS(build_cinn_pass, BuildCinnPass);
