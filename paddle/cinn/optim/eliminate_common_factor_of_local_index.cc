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
using ir::Expr;

class GatherLocalIndexAndProhibitedLocalVarVisitor
    : public ir::IRMutator<>,
      public ir::stmt::StmtVisitor<> {
 public:
  void operator()(ir::stmt::BlockRef func_body) { VisitBlock(func_body); }

  const std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>&
  local_var_to_indexes() const {
    return local_var_to_indexes_;
  }

  const std::unordered_set<std::string>& prohibited_local_vars() const {
    return prohibited_local_vars_;
  }

 private:
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

  void Visit(const Expr& expr) {
    Expr expr_ = expr;
    ir::IRMutator<>::Visit(&expr_, &expr_);
  }

  void VisitStmt(const ir::stmt::Store& stmt) override {
    Visit(stmt->value());

    if (!stmt->tensor().as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (stmt->tensor().as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      local_var_to_indexes_[stmt->tensor().as_tensor_ref()->buffer->name]
          .push_back(stmt->indices());

      if (stmt->value().As<ir::Call>()) {
        const std::string& local_var_name =
            stmt->tensor().as_tensor_ref()->buffer->name;
        const std::string& call_name = stmt->value().As<ir::Call>()->name;
        if (cinn::utils::GetProhibitScheduleExternalFuncNames().count(
                call_name) > 0) {
          prohibited_local_vars_.insert(local_var_name);
        }
      }
    }
  }

  void VisitStmt(const ir::stmt::IfThenElse& stmt) override {
    Visit(stmt->condition());
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(const ir::stmt::Schedule& stmt) override {
    for (const Expr& value : stmt->iter_values()) {
      Visit(value);
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(const ir::stmt::For& stmt) override {
    Visit(stmt->min());
    Visit(stmt->extent());
    VisitBlock(stmt->body());
  }

  void VisitStmt(const ir::stmt::Alloc& stmt) override {
    for (const Expr& extent : stmt->extents()) {
      Visit(extent);
    }
    if (stmt->condition().defined()) {
      Visit(stmt->condition());
    }
    if (stmt->body().defined()) {
      Visit(stmt->body());
    }
  }

  void VisitStmt(const ir::stmt::Evaluate& stmt) override {
    Visit(stmt->value());
  }

  void VisitStmt(const ir::stmt::Free& stmt) override {
    Visit(stmt->destination());
  }

  void VisitStmt(const ir::stmt::Let& stmt) override { Visit(stmt->body()); }

 private:
  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
      local_var_to_indexes_;
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
CollectLocalVarToIndexes(ir::stmt::BlockRef func_body) {
  GatherLocalIndexAndProhibitedLocalVarVisitor gather;
  gather(func_body);

  return EraseProhibitedLocalVar(gather.local_var_to_indexes(),
                                 gather.prohibited_local_vars());
}

int ExtractMulNumberFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = optim::ArithSimplify(expr);
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
  PADDLE_THROW(::common::errors::Fatal("Dead code"));
}

int ExtractAddNumberFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = optim::ArithSimplify(expr);
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
  PADDLE_THROW(::common::errors::Fatal("Dead code"));
}

int gcd(int a, int b) {
  if (b == 0) {
    return a == 0 ? 1 : a;
  }
  return gcd(b, a % b);
}

ir::Expr ExtractSymbolicFromExpr(const ir::Expr& expr) {
  ir::Expr simplied_expr = optim::ArithSimplify(expr);
  if (simplied_expr.is_constant()) {
    return ir::Expr(0);
  } else if (expr.As<ir::_Var_>()) {
    auto var = expr.As<ir::_Var_>();
    if (var->is_symbolic_constant) {
      VLOG(6) << "Extract symbolic constant, name = " << var->name;
      return ir::ir_utils::IRCopy(expr);
    }
    return ir::Expr(0);
  } else {
    VLOG(6) << "Not supported for calculating symbolic, expr = " << expr;
    return ir::Expr(0);
  }
  PADDLE_THROW(::common::errors::Fatal(
      "Dead code. Fail to extract symbolic from expression."));
}

class Gcd {};
class Offset {};
class Symbolic {};

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
      return optim::ArithSimplify(ir::Div::Make(expr, factor));
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
      return optim::ArithSimplify(ir::Sub::Make(expr, factor));
    }
    return expr;
  }
};

const ir::Expr CommonFactorTrait<Offset>::unit = ir::Expr(0);

template <>
struct CommonFactorTrait<Symbolic> {
  static const ir::Expr unit;

  static ir::Expr Calculate(const ir::Expr& expr1, const ir::Expr& expr2) {
    auto IsSymbolicNotEqual = [&](const ir::Expr& expr1,
                                  const ir::Expr& expr2) -> bool {
      return optim::ArithSimplify(
                 ir::Sub::Make(ExtractSymbolicFromExpr(expr1),
                               ExtractSymbolicFromExpr(expr2))) != ir::Expr(0);
    };
    if (IsSymbolicNotEqual(expr1, expr2)) {
      return ir::Expr(0);
    }
    return ExtractSymbolicFromExpr(expr1);
  }

  static ir::Expr Simplify(const ir::Expr& expr, const ir::Expr& factor) {
    if (factor != unit) {
      return optim::ArithSimplify(ir::Sub::Make(expr, factor));
    }
    return expr;
  }
};

const ir::Expr CommonFactorTrait<Symbolic>::unit = ir::Expr(0);

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
  PADDLE_ENFORCE_GE(
      indexes.size(),
      2,
      ::common::errors::InvalidArgument(
          "We should guarantee indexes.size() >= 2, because local variable "
          "should at least load and store once. "));
  for (std::size_t i = 1; i < indexes.size(); ++i) {
    // NOTE(Hongyu Jia): Ideally, we can guarantee the size of indexes are
    // equal However, some unit tests (e.g. test_resnet_cinn,
    // test_instance_norm_op are still running with the deprecated
    // OpScheduler, and the ir::Expr will break this guarantee after
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
class EliminateCommonFactorVisitor : public ir::IRMutator<>,
                                     public ir::stmt::StmtMutator<> {
 public:
  EliminateCommonFactorVisitor(
      const std::unordered_map<std::string, std::vector<ir::Expr>>&
          local_var_to_common_factor)
      : local_var_to_common_factor_(local_var_to_common_factor) {}

  void operator()(ir::stmt::BlockRef func_body) { VisitBlock(func_body); }

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

  void Visit(const Expr& expr) {
    Expr expr_ = expr;
    ir::IRMutator<>::Visit(&expr_, &expr_);
  }

  void VisitStmt(ir::stmt::Store stmt) override {
    Visit(stmt->value());
    const auto& store_buffer = stmt->tensor().as_tensor_ref()->buffer;

    if (!store_buffer.defined()) {
      return;
    }

    if (store_buffer->memory_type == ir::MemoryType::GPULocal) {
      if (local_var_to_common_factor_.count(store_buffer->name) == 0) {
        return;
      }
      const auto& common_factors =
          local_var_to_common_factor_.at(store_buffer->name);
      for (std::size_t i = 0; i < stmt->indices().size(); ++i) {
        std::vector<Expr> new_indices = stmt->indices();
        new_indices[i] =
            CommonFactorTrait<Op>::Simplify(new_indices[i], common_factors[i]);
        stmt->set_indices(new_indices);
      }
    }
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    Visit(stmt->condition());
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    for (const Expr& value : stmt->iter_values()) {
      Visit(value);
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(ir::stmt::For stmt) override {
    Visit(stmt->min());
    Visit(stmt->extent());
    VisitBlock(stmt->body());
  }

  void VisitStmt(ir::stmt::Alloc stmt) override {
    for (const Expr& extent : stmt->extents()) {
      Visit(extent);
    }
    if (stmt->condition().defined()) {
      Visit(stmt->condition());
    }
    if (stmt->body().defined()) {
      Visit(stmt->body());
    }
  }

  void VisitStmt(ir::stmt::Evaluate stmt) override { Visit(stmt->value()); }

  void VisitStmt(ir::stmt::Free stmt) override { Visit(stmt->destination()); }

  void VisitStmt(ir::stmt::Let stmt) override { Visit(stmt->body()); }

 private:
  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_common_factor_;
};

}  // namespace

// Eliminate common factors from local indices in a function's body.
// If applied to various statement blocks, this may incorrectly simplify
// distinct local buffer indices across different statement blocks to the same
// value.
template <typename Op>
void EliminateCommonFactorHelper(ir::stmt::BlockRef func_body) {
  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
      local_var_to_indexes = CollectLocalVarToIndexes(func_body);
  std::unordered_map<std::string, std::vector<ir::Expr>>
      local_var_to_common_factor =
          CalculateLocalVarCommonFactor<Op>(local_var_to_indexes);
  for (const auto& [local_var, common_factor] : local_var_to_common_factor) {
    auto index = local_var_to_indexes.at(local_var);
    for (std::size_t i = 0; i < index.size(); ++i) {
    }
  }
  EliminateCommonFactorVisitor<Op> eliminate_common_factor_visitor(
      local_var_to_common_factor);
  eliminate_common_factor_visitor(func_body);
}

class TransformLocalIndicesVisitor : public ir::IRMutator<>,
                                     public ir::stmt::StmtMutator<> {
 public:
  void operator()(ir::stmt::BlockRef func_body) { VisitBlock(func_body); }

 private:
  template <typename OpType>
  void ExtractIterHelper(
      const ir::Expr& expr,
      std::unordered_map<std::string, ir::Expr>* name_to_iter) {
    const auto op = expr.As<OpType>();
    ExtractIterFromIndice(op->a(), name_to_iter);
    ExtractIterFromIndice(op->b(), name_to_iter);
  }

  void ExtractIterFromIndice(
      const ir::Expr& expr,
      std::unordered_map<std::string, ir::Expr>* name_to_iter) {
    if (expr.As<ir::_Var_>()) {
      const auto var = expr.As<ir::_Var_>();
      if (name_to_iter->count(var->name) == 0) {
        (*name_to_iter)[var->name] = expr;
      }
    } else if (expr.As<ir::Add>()) {
      ExtractIterHelper<ir::Add>(expr, name_to_iter);
    } else if (expr.As<ir::Sub>()) {
      ExtractIterHelper<ir::Sub>(expr, name_to_iter);
    } else if (expr.As<ir::Mul>()) {
      ExtractIterHelper<ir::Mul>(expr, name_to_iter);
    } else if (expr.As<ir::Div>()) {
      ExtractIterHelper<ir::Div>(expr, name_to_iter);
    } else if (expr.As<ir::Mod>()) {
      ExtractIterHelper<ir::Mod>(expr, name_to_iter);
    } else {
      VLOG(4) << "Not support for extract iter: \n" << expr;
      return;
    }
    return;
  }

  std::vector<ir::Expr> ConvertIndicesToIters(
      const std::vector<ir::Expr>& indices) {
    auto CopyIndiceItersToLocalBuffer =
        [&](const std::unordered_map<std::string, ir::Expr>& name_to_iter,
            const std::vector<ir::Expr>& indices) -> std::vector<ir::Expr> {
      std::vector<ir::Expr> local_buffer_iters;
      for (std::size_t i = 0; i < loop_vars_.size(); ++i) {
        VLOG(6) << "loop var name: " << loop_vars_[i]->name;
        if (name_to_iter.count(loop_vars_[i]->name) > 0) {
          local_buffer_iters.push_back(name_to_iter.at(loop_vars_[i]->name));
        }
      }

      while (local_buffer_iters.size() < indices.size()) {
        local_buffer_iters.insert(local_buffer_iters.begin(), ir::Expr(0));
      }
      return local_buffer_iters;
    };

    std::unordered_map<std::string, ir::Expr> name_to_iter;
    for (const auto& index : indices) {
      ExtractIterFromIndice(index, &name_to_iter);
      VLOG(6) << "extract iter: " << index
              << " iter_set size: " << name_to_iter.size();
    }
    return CopyIndiceItersToLocalBuffer(name_to_iter, indices);
  }

  void Visit(const ir::Load* op, ir::Expr* expr) override {
    auto load = expr->As<ir::Load>();
    if (load->tensor.as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      load->indices = ConvertIndicesToIters(load->indices);
    }
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const Expr& expr) {
    Expr expr_ = expr;
    ir::IRMutator<>::Visit(&expr_, &expr_);
  }

  void VisitStmt(ir::stmt::Store stmt) override {
    if (stmt->tensor().as_tensor_ref()->buffer->memory_type ==
        ir::MemoryType::GPULocal) {
      stmt->set_indices(ConvertIndicesToIters(stmt->indices()));
    }
    Visit(stmt->value());
  }

  void VisitStmt(ir::stmt::IfThenElse stmt) override {
    Visit(stmt->condition());
    VisitBlock(stmt->true_case());
    if (stmt->false_case().defined()) {
      VisitBlock(stmt->false_case());
    }
  }

  void VisitStmt(ir::stmt::Schedule stmt) override {
    for (const Expr& value : stmt->iter_values()) {
      Visit(value);
    }
    VisitBlock(stmt->body());
  }

  void VisitStmt(ir::stmt::For stmt) override {
    Visit(stmt->min());
    Visit(stmt->extent());
    loop_vars_.push_back(stmt->loop_var());
    VisitBlock(stmt->body());
    loop_vars_.pop_back();
  }

  void VisitStmt(ir::stmt::Alloc stmt) override {
    for (const Expr& extent : stmt->extents()) {
      Visit(extent);
    }
    if (stmt->condition().defined()) {
      Visit(stmt->condition());
    }
    if (stmt->body().defined()) {
      Visit(stmt->body());
    }
  }

  void VisitStmt(ir::stmt::Evaluate stmt) override { Visit(stmt->value()); }

  void VisitStmt(ir::stmt::Free stmt) override { Visit(stmt->destination()); }

  void VisitStmt(ir::stmt::Let stmt) override { Visit(stmt->body()); }

 private:
  std::vector<ir::Var> loop_vars_;
};

void TransformLocalIndicesToIters(ir::stmt::BlockRef func_body) {
  TransformLocalIndicesVisitor transform_local_indices_visitor;
  transform_local_indices_visitor(func_body);
}

void EliminateCommonFactorOfLocalIndex(ir::stmt::BlockRef func_body) {
  VLOG(4) << "Before EliminateCommonFactorOfLocalIndex, func_body = \n"
          << func_body;
  EliminateCommonFactorHelper<Gcd>(func_body);
  EliminateCommonFactorHelper<Offset>(func_body);
  EliminateCommonFactorHelper<Symbolic>(func_body);

  TransformLocalIndicesToIters(func_body);

  VLOG(4) << "After EliminateCommonFactorOfLocalIndex, func_body = \n"
          << func_body;
}

}  // namespace optim
}  // namespace cinn
