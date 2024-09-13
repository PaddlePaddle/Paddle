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

#include "paddle/cinn/optim/merge_block_utils.h"

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace optim {

namespace {

struct ForInfoAnalyzer : public ir::IRMutator<Expr*> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  std::vector<std::vector<const ir::For*>> GetInnerForList() {
    std::vector<std::vector<const ir::For*>> inner_for_list;
    for (const auto& [level, for_list] : level_to_for_list_) {
      inner_for_list.push_back(for_list);
    }
    return inner_for_list;
  }

 private:
  void Visit(const ir::For* op, ir::Expr* expr) override {
    auto* node = expr->As<ir::For>();
    if (level_to_for_list_.count(current_for_level_) == 0) {
      level_to_for_list_[current_for_level_] = {node};
    } else {
      level_to_for_list_[current_for_level_].push_back(node);
    }

    ++current_for_level_;
    ir::IRMutator<>::Visit(op, expr);
    --current_for_level_;
  }

  int current_for_level_ = 0;
  std::unordered_map<int, std::vector<const ir::For*>> level_to_for_list_;
};

}  // namespace

bool CanMergeBlocks(const ir::For* first,
                    const ir::For* second,
                    const ForEqualFunc& IsEqual) {
  auto Get = [&](ir::Expr* expr) -> std::vector<std::vector<const ir::For*>> {
    ForInfoAnalyzer for_info_analyzer;
    for_info_analyzer(expr);
    return for_info_analyzer.GetInnerForList();
  };
  ir::Expr first_expr = Expr(const_cast<ir::For*>(first));
  ir::Expr second_expr = Expr(const_cast<ir::For*>(second));
  const auto first_inner_for_list = Get(&first_expr);
  const auto second_inner_for_list = Get(&second_expr);
  return IsEqual(first_inner_for_list, second_inner_for_list);
}

}  // namespace optim
}  // namespace cinn
