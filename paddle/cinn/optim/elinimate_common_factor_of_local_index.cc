// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/elinimate_common_factor_of_local_index.h"

#include <unordered_map>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/external_func_names.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {
namespace {

class GatherLocalIndexVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  const std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>&
  local_var_to_indexes() const {
    return local_var_to_indexes_;
  }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto store = expr->As<ir::Store>();

    ir::IRMutator<>::Visit(op, expr);
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (store->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      local_var_to_indexes_[store->tensor.as_tensor_ref()->buffer->name]
          .push_back(store->indices);
    }
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto load = expr->As<ir::Load>();

    if (load->is_addr_scalar()) {
      return;
    }
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      local_var_to_indexes_[load->tensor.as_tensor_ref()->buffer->name]
          .push_back(load->indices);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
      local_var_to_indexes_;
};

class GatherProhibitedLocalVarVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  const std::unordered_set<std::string>& prohibited_local_vars() const {
    return prohibited_local_vars_;
  }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto store = expr->As<ir::Store>();

    ir::IRMutator<>::Visit(op, expr);
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }
    if (store->tensor.as_tensor_ref()->buffer->memory_type !=
        ir::MemoryType::GPULocal) {
      return;
    }
    const auto& local_var_name = store->tensor.as_tensor_ref()->buffer->name;
    if (store->value.As<ir::Call>()) {
      const auto& call_name = store->value.As<ir::Call>()->name;
      if (cinn::utils::kProhibitScheduleExternalFuncNames.count(call_name) >
          0) {
        prohibited_local_vars_.insert(local_var_name);
      }
    }
  }

  std::unordered_set<std::string> prohibited_local_vars_;
};

std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
EraseProhibitedLocalVar(
    const std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>&
        local_var_to_indexes,
    const std::unordered_set<std::string>& prohibited_local_vars) {
  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>> ret{};
  for (const auto& [local_var, indexes] : local_var_to_indexes) {
    if (prohibited_local_vars.count(local_var) == 0) {
      ret[local_var] = indexes;
    }
  }
  return ret;
}

std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
CollectLocalVarToIndexes(ir::Expr* expr) {
  GatherLocalIndexVisitor gather_local_index_visitor;
  gather_local_index_visitor(expr);

  GatherProhibitedLocalVarVisitor gather_prohibited_local_var_visitor;
  gather_prohibited_local_var_visitor(expr);

  return EraseProhibitedLocalVar(
      gather_local_index_visitor.local_var_to_indexes(),
      gather_prohibited_local_var_visitor.prohibited_local_vars());
}

template <typename DoEachT>
void VisitEachRowExpr(const std::vector<std::vector<ir::Expr>>& indexes,
                      std::size_t var_idx,
                      DoEachT&& DoEach) {
  for (std::size_t i = 0; i < indexes.size(); ++i) {
    DoEach(indexes[i][var_idx]);
  }
}

int ExtractNumberFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = cinn::common::AutoSimplify(expr);
  if (simplied_expr.is_constant()) {
    return static_cast<int>(simplied_expr.get_constant());
  } else if (expr.As<ir::Mul>()) {
    auto mul = expr.As<ir::Mul>();
    return std::max(ExtractNumberFromExpr(mul->a()),
                    ExtractNumberFromExpr(mul->b()));
  } else {
    VLOG(6) << "Not supported for calculating gcd, expr = " << expr;
    return 1;
  }
  LOG(FATAL) << "Dead code";
}

int gcd(int a, int b) {
  if (b == 0) {
    return a;
  }
  return gcd(b, a % b);
}

// Note (Hongyu Jia): Currently, we only calculates gcd of int factors.
ir::Expr CalculateGcdForExprPair(const ir::Expr& expr1, const ir::Expr& expr2) {
  return ir::Expr(
      gcd(ExtractNumberFromExpr(expr1), ExtractNumberFromExpr(expr2)));
}

std::vector<ir::Expr> CalculateIndexVectorGcd(
    const std::string& local_var,
    const std::vector<std::vector<ir::Expr>>& indexes) {
  CHECK_GE(indexes.size(), 2)
      << "We should guarantee indexes.size() >= 2, because local variable "
      << local_var << "should at least load and store once.";
  for (std::size_t i = 1; i < indexes.size(); ++i) {
    CHECK_EQ(indexes[0].size(), indexes[i].size())
        << "We should guarantee all index vectors have the same size.";
  }
  std::size_t var_index_size = indexes[0].size();
  std::vector<ir::Expr> gcd_indexes;
  for (std::size_t var_idx = 0; var_idx < var_index_size; ++var_idx) {
    std::optional<ir::Expr> gcd_expr;
    VisitEachRowExpr(indexes, var_idx, [&](const ir::Expr& expr) {
      if (gcd_expr.has_value()) {
        gcd_expr = CalculateGcdForExprPair(gcd_expr.value(), expr);
      } else {
        gcd_expr = expr;
      }
    });
    gcd_indexes.push_back(gcd_expr.value());
  }
  return gcd_indexes;
}

std::unordered_map<std::string, std::vector<ir::Expr>> CalculateLocalIndexGcd(
    const std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>&
        local_var_to_indexes) {
  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_gcd_factor;
  for (const auto& [local_var, indexes] : local_var_to_indexes) {
    local_var_to_gcd_factor[local_var] =
        CalculateIndexVectorGcd(local_var, indexes);
  }
  return local_var_to_gcd_factor;
}

class DivideGcdForLocalIndexVisitor : public ir::IRMutator<> {
 public:
  DivideGcdForLocalIndexVisitor(
      const std::unordered_map<std::string, std::vector<ir::Expr>>&
          local_var_to_gcd_factor)
      : local_var_to_gcd_factor_(local_var_to_gcd_factor) {}

  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto store = expr->As<ir::Store>();

    ir::IRMutator<>::Visit(op, expr);
    const auto& store_buffer = store->tensor.as_tensor_ref()->buffer;
    if (!store_buffer.defined()) {
      return;
    }

    if (store_buffer->memory_type == ir::MemoryType::GPULocal) {
      if (local_var_to_gcd_factor_.count(store_buffer->name) == 0) {
        return;
      }
      const auto& gcd_factors = local_var_to_gcd_factor_.at(store_buffer->name);
      CHECK(store->indices.size() == gcd_factors.size())
          << "Store index size should be equal to gcd factor size.";
      for (std::size_t i = 0; i < store->indices.size(); ++i) {
        if (gcd_factors[i] != ir::Expr(0)) {
          store->indices[i] = cinn::common::AutoSimplify(
              ir::Div::Make(store->indices[i], gcd_factors[i]));
        }
      }
    }
  }

  void Visit(const ir::Load* op, Expr* expr) override {
    auto load = expr->As<ir::Load>();

    if (load->is_addr_scalar()) {
      return;
    }
    const auto& load_buffer = load->tensor.as_tensor_ref()->buffer;
    if (!load_buffer.defined()) {
      return;
    }

    if (load_buffer->memory_type == ir::MemoryType::GPULocal) {
      if (local_var_to_gcd_factor_.count(load_buffer->name) == 0) {
        return;
      }
      const auto& gcd_factors = local_var_to_gcd_factor_.at(load_buffer->name);
      CHECK(load->indices.size() == gcd_factors.size())
          << "Store index size should be equal to gcd factor size.";
      for (std::size_t i = 0; i < load->indices.size(); ++i) {
        if (gcd_factors[i] != ir::Expr(0)) {
          load->indices[i] = cinn::common::AutoSimplify(
              ir::Div::Make(load->indices[i], gcd_factors[i]));
        }
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }
  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_gcd_factor_;
};

}  // namespace

void ElinimateCommonFactorOfLocalIndex(ir::Expr* expr) {
  VLOG(2) << "Before ElinimateCommonFactorOfLocalIndex, Expr = \n" << *expr;

  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
      local_var_to_indexes = CollectLocalVarToIndexes(expr);

  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_gcd_factor = CalculateLocalIndexGcd(local_var_to_indexes);

  DivideGcdForLocalIndexVisitor divide_gcd_for_local_index_visitor(
      local_var_to_gcd_factor);
  divide_gcd_for_local_index_visitor(expr);

  VLOG(2) << "After ElinimateCommonFactorOfLocalIndex, Expr = \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
