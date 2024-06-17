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

#include "paddle/cinn/optim/eliminate_common_factor_of_local_index.h"

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
      if (cinn::utils::GetProhibitScheduleExternalFuncNames().count(call_name) >
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

int ExtractMulNumberFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = cinn::common::AutoSimplify(expr);
  if (simplied_expr.is_constant()) {
    return static_cast<int>(simplied_expr.get_constant());
  } else if (expr.As<ir::Mul>()) {
    auto mul = expr.As<ir::Mul>();
    return ExtractMulNumberFromExpr(mul->a()) *
           ExtractMulNumberFromExpr(mul->b());
  } else {
    VLOG(6) << "Not supported for calculating gcd, expr = " << expr;
    return 1;
  }
  PADDLE_THROW(phi::errors::Fatal("Dead code"));
}

int ExtractAddNumberFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = cinn::common::AutoSimplify(expr);
  if (simplied_expr.is_constant()) {
    return static_cast<int>(simplied_expr.get_constant());
  } else if (expr.As<ir::Add>()) {
    auto add = expr.As<ir::Add>();
    return ExtractAddNumberFromExpr(add->a()) +
           ExtractAddNumberFromExpr(add->b());
  } else {
    VLOG(6) << "Not supported for calculating offset, expr = " << expr;
    return 0;
  }
  PADDLE_THROW(phi::errors::Fatal("Dead code"));
}

int gcd(int a, int b) {
  if (b == 0) {
    return a == 0 ? 1 : a;
  }
  return gcd(b, a % b);
}

class Gcd {};
class Offset {};

template <typename Op>
struct CommonFactorTrait;

template <>
struct CommonFactorTrait<Gcd> {
  static const ir::Expr unit;

  // Note (Hongyu Jia): Currently, we only calculates gcd of int factors.
  static ir::Expr Calculate(const ir::Expr& expr1, const ir::Expr& expr2) {
    return ir::Expr(
        gcd(ExtractMulNumberFromExpr(expr1), ExtractMulNumberFromExpr(expr2)));
  }

  static ir::Expr Simplify(const ir::Expr& expr, const ir::Expr& factor) {
    if (factor != unit) {
      return cinn::common::AutoSimplify(ir::Div::Make(expr, factor));
    }
    return expr;
  }
};

const ir::Expr CommonFactorTrait<Gcd>::unit = ir::Expr(1);

template <>
struct CommonFactorTrait<Offset> {
  static const ir::Expr unit;

  static ir::Expr Calculate(const ir::Expr& expr1, const ir::Expr& expr2) {
    return ir::Expr(std::min(ExtractAddNumberFromExpr(expr1),
                             ExtractAddNumberFromExpr(expr2)));
  }

  static ir::Expr Simplify(const ir::Expr& expr, const ir::Expr& factor) {
    if (factor != unit) {
      return cinn::common::AutoSimplify(ir::Sub::Make(expr, factor));
    }
    return expr;
  }
};

const ir::Expr CommonFactorTrait<Offset>::unit = ir::Expr(0);

template <typename DoEachT>
void VisitEachRowExpr(const std::vector<std::vector<ir::Expr>>& indexes,
                      std::size_t var_idx,
                      DoEachT&& DoEach) {
  for (std::size_t i = 0; i < indexes.size(); ++i) {
    DoEach(indexes[i][var_idx]);
  }
}

template <typename Op>
std::vector<ir::Expr> CalculateIndexCommonFactor(
    const std::string& local_var,
    const std::vector<std::vector<ir::Expr>>& indexes) {
  CHECK_GE(indexes.size(), 2)
      << "We should guarantee indexes.size() >= 2, because local variable "
      << local_var << " should at least load and store once.";
  for (std::size_t i = 1; i < indexes.size(); ++i) {
    // NOTE(Hongyu Jia): Ideally, we can guarantee the size of indexes are equal
    // under flags FLAGS_cinn_new_group_scheduler=1 and
    // FLAGS_cinn_bucket_compile=1. However, some unit tests (e.g.
    // test_resnet_cinn, test_instance_norm_op) are still running with the
    // deprecated OpScheduler, and the ir::Expr will break this guarantee after
    // IRGpuScheduleBlockReduce function. So we have to relax the restriction
    // here.
    if (indexes[i].size() != indexes[0].size()) {
      LOG(WARNING)
          << "Not supported for calculating common factor, local var = "
          << local_var;
      return std::vector<ir::Expr>(
          std::max(indexes[0].size(), indexes[i].size()),
          CommonFactorTrait<Op>::unit);
    }
  }
  std::size_t var_index_size = indexes[0].size();
  std::vector<ir::Expr> common_factor_indexes;
  for (std::size_t var_idx = 0; var_idx < var_index_size; ++var_idx) {
    std::optional<ir::Expr> common_factor;
    VisitEachRowExpr(indexes, var_idx, [&](const ir::Expr& expr) {
      if (common_factor.has_value()) {
        common_factor =
            CommonFactorTrait<Op>::Calculate(common_factor.value(), expr);
      } else {
        common_factor = expr;
      }
    });
    common_factor_indexes.push_back(common_factor.value());
  }
  return common_factor_indexes;
}

template <typename Op>
std::unordered_map<std::string, std::vector<ir::Expr>>
CalculateLocalVarCommonFactor(
    const std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>&
        local_var_to_indexes) {
  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_common_factor;
  for (const auto& [local_var, indexes] : local_var_to_indexes) {
    local_var_to_common_factor[local_var] =
        CalculateIndexCommonFactor<Op>(local_var, indexes);
  }
  return local_var_to_common_factor;
}

template <typename Op>
class EliminateCommonFactorVisitor : public ir::IRMutator<> {
 public:
  EliminateCommonFactorVisitor(
      const std::unordered_map<std::string, std::vector<ir::Expr>>&
          local_var_to_common_factor)
      : local_var_to_common_factor_(local_var_to_common_factor) {}

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
      if (local_var_to_common_factor_.count(store_buffer->name) == 0) {
        return;
      }
      const auto& common_factors =
          local_var_to_common_factor_.at(store_buffer->name);
      for (std::size_t i = 0; i < store->indices.size(); ++i) {
        store->indices[i] = CommonFactorTrait<Op>::Simplify(store->indices[i],
                                                            common_factors[i]);
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
      if (local_var_to_common_factor_.count(load_buffer->name) == 0) {
        return;
      }
      const auto& common_factors =
          local_var_to_common_factor_.at(load_buffer->name);
      for (std::size_t i = 0; i < load->indices.size(); ++i) {
        load->indices[i] = CommonFactorTrait<Op>::Simplify(load->indices[i],
                                                           common_factors[i]);
      }
    }
    ir::IRMutator<>::Visit(op, expr);
  }
  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_common_factor_;
};

}  // namespace

template <typename Op>
void EliminateCommonFactorHelper(ir::Expr* expr) {
  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
      local_var_to_indexes = CollectLocalVarToIndexes(expr);
  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_common_factor =
          CalculateLocalVarCommonFactor<Op>(local_var_to_indexes);
  EliminateCommonFactorVisitor<Op> eliminate_common_factor_visitor(
      local_var_to_common_factor);
  eliminate_common_factor_visitor(expr);
}

void EliminateCommonFactorOfLocalIndex(ir::Expr* expr) {
  VLOG(4) << "Before EliminateCommonFactorOfLocalIndex, Expr = \n" << *expr;
  EliminateCommonFactorHelper<Gcd>(expr);
  EliminateCommonFactorHelper<Offset>(expr);
  VLOG(4) << "After EliminateCommonFactorOfLocalIndex, Expr = \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
