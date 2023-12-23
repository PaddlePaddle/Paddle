// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/search_space/search_state.h"

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

SearchState::SearchState(ir::IRSchedule ir_sch,
                         float cost,
                         const std::vector<AutoGenRule*>& rules)
    : cinn::common::Shared<_SearchState_>(
          cinn::common::make_shared<_SearchState_>()) {
  auto* state = get();
  state->ir_schedule = std::move(ir_sch);
  state->applicable_rules = rules;
  state->predicted_cost = cost;
}

SearchState SearchState::Copy() const {
  return SearchState((*this)->ir_schedule, (*this)->predicted_cost, {});
}

std::string _SearchState_::DebugString() const {
  const auto& exprs = ir_schedule.GetModule().GetExprs();
  std::stringstream module_stream;
  for (auto i = 0; i < exprs.size(); ++i) {
    module_stream << "Expr " << i << " {\n"
                  << exprs.at(i) << "\n}  // end Expr";
  }

  const char* fmt_str = R"ROC(
ModuleExpr {
%s
} // end ModuleExpr
ScheduleDesc {
%s
} // end ScheduleDesc
predicted_cost: %f)ROC";

  return utils::StringFormat(fmt_str,
                             module_stream.str().c_str(),
                             ir_schedule.GetTraceDesc().DebugString().c_str(),
                             predicted_cost);
}

bool operator<(const SearchState& left, const SearchState& right) {
  return left->predicted_cost < right->predicted_cost;
}

// Visit every node by expanding all of their fields in dfs order
class DfsWithExprsFields : public ir::IRVisitorRequireReImpl<void> {
 protected:
#define __m(t__)                          \
  void Visit(const ir::t__* x) override { \
    for (auto* n : x->expr_fields()) {    \
      if (n->defined()) {                 \
        Visit(n);                         \
      }                                   \
    }                                     \
  }

  NODETY_FORALL(__m)
#undef __m

  void Visit(const Expr* expr) override { IRVisitorRequireReImpl::Visit(expr); }
};

// Generate a reduce hash of a AST tree by combining hash of each AST node
class IrNodesStructuralHash : public DfsWithExprsFields {
 public:
  explicit IrNodesStructuralHash(size_t init_key) : hash_key_(init_key) {}
  size_t operator()(const Expr* expr) {
    Visit(expr);
    return hash_key_;
  }

  void Visit(const Expr* expr) override {
    static decltype(ir::kIrNodeTyReprs) Node2Name = ir::kIrNodeTyReprs;
    if (!expr->defined()) return;
    auto type_code = static_cast<IrNodeTyUnderlyingType>(expr->node_type());
    hash_key_ = utils::HashCombine(hash_key_, type_code);
    DfsWithExprsFields::Visit(expr);
  }

 private:
  void Visit(const ir::_Tensor_* x) override {
    for (auto& e : x->shape) {
      Visit(&e);
    }
    DfsWithExprsFields::Visit(x->buffer.As<ir::_Buffer_>());
  }

  using IrNodeTyUnderlyingType = std::underlying_type<ir::IrNodeTy>::type;
  size_t hash_key_;
};

size_t SearchStateHash::operator()(const SearchState& s) const {
  size_t hash_key = 0;
  const auto& exprs = s->ir_schedule.GetModule().GetExprs();
  for (auto&& expr : exprs) {
    hash_key = IrNodesStructuralHash(hash_key)(&expr);
  }
  return hash_key;
}

bool SearchStateEqual::operator()(const SearchState& lhs,
                                  const SearchState& rhs) const {
  const auto& lhs_exprs = lhs->ir_schedule.GetModule().GetExprs();
  const auto& rhs_exprs = rhs->ir_schedule.GetModule().GetExprs();
  // compare exprs size firstly
  if (lhs_exprs.size() != rhs_exprs.size()) return false;

  // compare every expr one by one with ir::ir_utils::IrEqualVisitor
  for (int i = 0; i < lhs_exprs.size(); ++i) {
    if (!ir::ir_utils::IRCompare(lhs_exprs[i], rhs_exprs[i], true))
      return false;
  }
  return true;
}

std::string JoinStatesDebugString(const std::string& title,
                                  const std::vector<SearchState>& states,
                                  bool verbose) {
  std::stringstream ss;
  ss << title << " states size:" << states.size() << "\n";
  SearchStateHash state_hasher;
  for (size_t i = 0; i < states.size(); ++i) {
    uint64_t hash_key = state_hasher(states[i]);
    if (verbose) {
      ss << "\tState-" << i << " hash:" << hash_key << "\t content:------>"
         << states[i]->DebugString() << "\n<------";
    } else {
      ss << "\tState-" << i << " hash:" << hash_key << "\n";
    }
  }
  return std::move(*ss.rdbuf()).str();
}

}  // namespace auto_schedule
}  // namespace cinn
