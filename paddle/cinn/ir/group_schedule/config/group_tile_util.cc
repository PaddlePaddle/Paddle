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

#include "paddle/cinn/ir/group_schedule/config/group_tile_util.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"
#include "paddle/cinn/ir/schedule/impl/ir_schedule.h"
namespace cinn {

using hlir::framework::pir::trivial_fusion_detail::GetAllForIters;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ChildScheduleBlockRealizes;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ScheduleBlockRealizeIsNotInit;

namespace ir {
namespace {

struct VarReplacer : public ir::IRMutator<ir::Expr*> {
  std::unordered_set<ir::Var> iter_vars;
  ir::Var inspecting_var;

  explicit VarReplacer(const std::vector<ir::Var>& _iter_vars)
      : iter_vars(_iter_vars.begin(), _iter_vars.end()) {}

  virtual void Visit(const ir::_Var_* op, ir::Expr* expr) {
    ir::Var var = op->Copy().as_var_ref();
    if (inspecting_var.defined() && var == inspecting_var) {
      *expr = ir::Expr(1);
    } else if (iter_vars.find(var) != iter_vars.end()) {
      *expr = ir::Expr(0);
    } else {
      // We can replace shape variables (e.g. S0) with any constant, and here
      // we just choose to replace them with 32.
      *expr = ir::Expr(32);
    }
  }
};

std::vector<ir::Expr> GetRValueLoads(ir::Expr expr_block) {
  ir::Expr store = analyzer::GetStoreOfSBlock(expr_block);
  auto* store_node = store.As<ir::Store>();
  return ir::ir_utils::CollectIRNodesInOrder(
      store_node->value, [&](const ir::Expr* x) {
        auto* load_node = x->As<ir::Load>();
        return load_node && load_node->tensor != store_node->tensor;
      });
}

std::vector<int64_t> GetVarStrides(ir::Expr load_offset,
                                   const std::vector<ir::Var>& iter_vars) {
  VarReplacer replacer(iter_vars);

  const auto Evaluate = [&](const ir::Var var) {
    ir::Expr expr = ir::ir_utils::IRCopy(load_offset);
    replacer.inspecting_var = var;
    replacer.IRMutator::Visit(&expr, &expr);
    ir::Expr res = optim::ArithSimplify(expr);
    if (res.is_constant()) {
      return res.as_int64();
    }
    return int64_t(0);
  };

  const int64_t base = Evaluate(ir::Var());

  std::vector<int64_t> strides;
  for (const auto& var : iter_vars) {
    int64_t stride = Evaluate(var) - base;
    strides.push_back(stride);
  }
  return strides;
}

ir::Expr GetLargestLoad(const std::vector<ir::Expr>& exprs) {
  common::cas_intervals_t var_intervals =
      common::CollectVarIntervalsOfExprs(exprs);
  common::SymbolicExprAnalyzer symbolic_expr_analyzer(var_intervals);

  const auto GetLoadSize = [](const ir::Expr& expr) {
    auto* load = expr.As<ir::Load>();
    auto* tensor = load->tensor.As<ir::_Tensor_>();
    if (tensor->shape.size() == 0) {
      return ir::Expr(1);
    }
    ir::Expr size = tensor->shape[0];
    for (size_t i = 1; i < tensor->shape.size(); i++) {
      size = size * tensor->shape[i];
    }
    return optim::ArithSimplify(size);
  };

  ir::Expr res = exprs[0];
  ir::Expr res_size = GetLoadSize(res);
  for (size_t i = 1; i < exprs.size(); i++) {
    ir::Expr cur_size = GetLoadSize(exprs[i]);
    std::optional<bool> gt = symbolic_expr_analyzer.ProveGT(cur_size, res_size);
    if (gt.has_value() && gt.value()) {
      res = exprs[i];
      res_size = cur_size;
    }
  }
  return res;
}

std::vector<ir::Expr> GetOpComputeBodyScheduleBlockRealizeExprSet(
    const ir::Expr& body) {
  using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ExprSetFinder;
  ExprSetFinder finder =
      ChildScheduleBlockRealizes * ExprSetFinder::GetIdentity();
  std::vector<ir::Expr> exprs = finder(body);
  return std::move(exprs);
}

bool ScheduleBlockRealizesShouldVectorizeCheck(
    const std::vector<ir::Expr>& exprs) {
  if (exprs.size() != 1) return false;

  ir::Expr expr_schedule_block_realize = exprs[0];
  bool is_reduce = ir::analyzer::IsReductionSBlock(expr_schedule_block_realize);
  if (is_reduce) return false;
  return true;
}

bool ScheduleBlockRealizeHasSpecialOp(
    Expr block, std::function<bool(const ir::Expr* e)>&& special_op_check) {
  bool has_special_op = false;
  ir::ir_utils::CollectIRNodesWithoutTensor(
      block,
      [&](const ir::Expr* expr) {
        if (special_op_check(expr)) {
          has_special_op = true;
          return true;
        }
        return false;
      },
      /* uniq_target = */ false);
  return has_special_op;
}

std::vector<bool> CollectTensorBroadcastAxis(const std::vector<Expr>& indices) {
  std::vector<bool> broadcast_axis(indices.size(), false);
  for (int i = 0; i < indices.size(); ++i) {
    ir::Expr index = indices[i];
    index = optim::ArithSimplify(index);
    if (index.is_constant() && index.get_constant() == 0) {
      broadcast_axis[i] = true;
    }
  }
  return broadcast_axis;
}

bool CheckBroadcastTensorIsContinuous(
    const std::string& tensor_name,
    const std::vector<Expr>& indices,
    const std::vector<ir::Var>& for_iters,
    const std::unordered_map<ir::Var, ir::Expr>& iter_var2value) {
  int loop_idx = 0;
  bool is_broadcast = false;
  for (int i = 0; i < indices.size(); ++i) {
    ir::Expr index = indices[i];
    index = optim::ArithSimplify(index);
    if (index.is_constant() && index.get_constant() == 0) {
      is_broadcast = true;
      continue;
    }

    if (!index.is_var()) return false;
    ir::Var iter_var = index.as_var_ref();
    if (!iter_var2value.count(iter_var)) {
      return false;
    }

    ir::Expr iter_value = iter_var2value.at(iter_var);

    if (!iter_value.as_var()) return false;
    for (; loop_idx < for_iters.size(); ++loop_idx) {
      if (for_iters[loop_idx] == iter_value.as_var_ref()) {
        break;
      }
    }

    if (loop_idx == for_iters.size()) {
      return false;
    }
  }

  if (is_broadcast || indices.size() < for_iters.size()) {
    return true;
  }
  return false;
}

// 检查常量是否是 2/4 的整数倍（满足 float2/float4 对齐）
bool IsAlignedConstant(int value, int alignment = 4) {
  return (value % alignment) == 0;
}

// 辅助函数：递归提取加法表达式的所有项
void ExtractAddTerms(const ir::Expr& expr, std::vector<ir::Expr>* terms) {
  if (expr.As<ir::Add>()) {
    ExtractAddTerms(expr.As<ir::Add>()->a(), terms);
    ExtractAddTerms(expr.As<ir::Add>()->b(), terms);
  } else {
    terms->push_back(expr);
  }
}

// 检查表达式是否是线性组合（如 j*C3 + k + C4）
bool IsLinearOffsetExpr(
    const ir::Expr& expr,
    const ir::Var& outer_iter,
    const ir::Var& inner_iter,
    const std::unordered_map<ir::Var, ir::Expr>& iter_var2value,
    bool *has_outer = nullptr,    // 是否找到 outer_iter（j）
    bool *has_inner = nullptr,    // 是否找到 inner_iter（k）
    bool *has_constant = nullptr, // 是否找到常量项（C4）
    int *outer_coeff = nullptr,       // outer_iter 的系数（C3）
    int *constant_term = nullptr     // 常量项的值（C4）
) {
  VLOG(6) << "YUHAN!!! IsLinearOffsetExpr? " << expr;
  VLOG(6) << "YUHAN!!! outer_iter " << outer_iter;
  VLOG(6) << "YUHAN!!! inner_iter " << inner_iter;

  // 1. 简化表达式
  std::vector<Var> replaced;
  std::vector<Expr> candidates;
  for (auto iter : iter_var2value) {
    replaced.push_back(iter.first);
    candidates.push_back(iter.second);
  }
  ir::Expr simplified = expr;
  ReplaceExpr(&simplified, replaced, candidates);
  VLOG(6) << "YUHAN!!! IsLinearOffsetExpr simplified " << simplified;
  simplified = optim::ArithSimplify(expr, IndexExpr::OptLevel::kLevel3);
  VLOG(6) << "YUHAN!!! IsLinearOffsetExpr simplified after ArithSimplify " << simplified;

  // 2. 提取所有加法项（支持嵌套加法结构）
  std::vector<ir::Expr> terms;
  ExtractAddTerms(simplified, &terms);  // 递归提取加法项

  for (const ir::Expr& term : terms) {
    VLOG(6) << "YUHAN!!! IsLinearOffsetExpr: work on term " << term;
    // 情况1：term 是 j*C3 乘法项）
    if (term.As<ir::Mul>() && term.As<ir::Mul>()->b().is_constant()) {
      ir::Expr var = term.As<ir::Mul>()->a();
      ir::Expr coeff = term.As<ir::Mul>()->b();
      VLOG(6) << "YUHAN!!! Condition 1 Found var: " << var << " YUHAN!!! "<< iter_var2value.count(var.as_var_ref()) << " YUHAN!!! " << outer_iter;
      if (var.is_var() && iter_var2value.count(var.as_var_ref())) {
        ir::Expr iter_value = iter_var2value.at(var.as_var_ref());
        if (iter_value.is_var() && iter_value.as_var_ref() == outer_iter) {
          *has_outer = true;
          *outer_coeff = coeff.get_constant();
          continue;
        }
      } else if (var.is_var() && !iter_var2value.count(var.as_var_ref()) && var.as_var_ref() == outer_iter) { // i0_43, i1_39 = axis.bind(i, ((j * 128) + j_0))
        VLOG(6) << "YUHAN!!! Condition 1 Found var new bind Condition: " << var;
        *has_outer = true;
        *outer_coeff = coeff.get_constant();
        continue;
      }
    }

    // 情况2：term 是 k（变量项）
    if (term.is_var() && iter_var2value.count(term.as_var_ref())) {
      ir::Expr iter_value = iter_var2value.at(term.as_var_ref());
      VLOG(6) << "YUHAN!!! Condition 2 Found iter_value: " << iter_value;
      if (iter_value.is_var() && iter_value.as_var_ref() == inner_iter) {
        *has_inner = true;
        continue;
      } else if (!iter_value.is_var()) {
        /* iter_value 不是变量，而是一个表达式，例如：
        serial for (i, 0ll, 32768ll)
        {
            serial for (j_1, 0, 16)
            {
                serial for (j_2, 0, 128)
                {
                    ScheduleBlock(var_45)
                    {
                        i0_88, i1_80 = axis.bind(i, ((j_1 * 128) + j_2))
                        {
                            var_45[i0_88, i1_80];
                        }
                    }
                }
            }
        }
        */
        VLOG(6) << "YUHAN!!! Condition 2 Found iter_value is an expr: " << iter_value;
        if (IsLinearOffsetExpr(iter_value, outer_iter, inner_iter, iter_var2value, has_outer, has_inner, has_constant, outer_coeff, constant_term)) {
          VLOG(6) << "YUHAN!!! Condition 2 Found iter_value is Linear combo: " << iter_value;
          continue;
        }
      }
    } else if (term.is_var() && !iter_var2value.count(term.as_var_ref()) && term.as_var_ref() == inner_iter) { // i0_43, i1_39 = axis.bind(i, ((j * 128) + j_0))
      VLOG(6) << "YUHAN!!! Condition 2 Found iter_value new bind Condition: " << term; // may be outer_iter!!
      *has_inner = true;
      continue;
    }

    // 情况3：term 是常量 C4
    if (term.is_constant()) {
      *has_constant = true;
      *constant_term = term.get_constant();
      VLOG(6) << "YUHAN!!! Condition 3 Found constant: " << term.get_constant();
      continue;
    }

    // 其他情况：非法项
    return false;
  }

  // // 必须包含 outer_iter 和 inner_iter
  // if (!has_outer || !has_inner) {
  //   VLOG(6) << "YUHAN!!! Missing outer or inner iter in linear combo";
  //   return false;
  // }

  // 调试输出
  VLOG(6) << "YUHAN!!! Found linear combo: "
          << *outer_coeff << "*" << outer_iter << " + " << inner_iter;
  if (*has_constant) {
    VLOG(6) << " + " << *constant_term;
  }

  return true;
}

bool CheckTensorIsContinuous(
    const std::vector<Expr>& indices,
    const std::vector<ir::Var>& for_iters,
    const std::unordered_map<ir::Var, ir::Expr>& iter_var2value) {
  VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous indices size is " << indices.size() << " " << for_iters.size();
  if (indices.size() == for_iters.size()) {
    for (int i = 0; i < indices.size(); ++i) {
      ir::Expr index = indices[i];
      std::vector<Var> replaced;
      std::vector<Expr> candidates;
      for (auto iter : iter_var2value) {
        replaced.push_back(iter.first);
        candidates.push_back(iter.second);
      }
      ReplaceExpr(&index, replaced, candidates);
      index = optim::ArithSimplify(index, IndexExpr::OptLevel::kLevel3);
      VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous index is " << index;
      if (!index.is_var()) {
        VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous index is not a var: " << index;
        return false;
      }
      if (for_iters[i] != index.as_var_ref()) {
        VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous for_iters[" << i << "] != index.as_var_ref(): " << for_iters[i] << " vs " << index.as_var_ref();
        return false;
      }
      // ir::Var iter_var = index.as_var_ref();
      // if (!iter_var2value.count(iter_var)) {
      //   VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous iter_var not found in iter_var2value: " << iter_var;
      //   return false;
      // }
      // ir::Expr iter_value = iter_var2value.at(iter_var);
      // if (!iter_value.as_var()) {
      //   VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous iter_value is not a var: " << iter_value;
      //   return false;
      // }
      // if (for_iters[i] != iter_value.as_var_ref()) {
      //   VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous for_iters[" << i << "] != iter_value.as_var_ref(): " << for_iters[i] << " vs " << iter_value.as_var_ref();
      //   return false;
      // }
    }
    return true;
  } else {
    /* After reshapeOp and Inline
    for i in range(C1):
      for j in range(C2):
        for k in range(C3):
            B[i][j][k] = A[i][C3*j + k] + A[i][C3*j + k + C4]
    */
    // 条件1：indices 必须比 for_iters 多1个（如 A[i][j*C3 +k] 对应循环 i,j,k）
    if (indices.size() != for_iters.size() - 1) {
      VLOG(6) << "YUHAN!! Indices/for_iters size mismatch for offset access";
      return false;
    }

    // 检查前 N-1 维是否直接对应循环变量（如 A[i][...] 中的 i）
    for (int i = 0; i < indices.size() - 1; ++i) {
      ir::Expr index = indices[i];
      std::vector<Var> replaced;
      std::vector<Expr> candidates;
      for (auto iter : iter_var2value) {
        replaced.push_back(iter.first);
        candidates.push_back(iter.second);
      }
      ReplaceExpr(&index, replaced, candidates);
      index = optim::ArithSimplify(index, IndexExpr::OptLevel::kLevel3);
      VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous index is " << index;
      if (!index.is_var()) {
        VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous index is not a var: " << index;
        return false;
      }
      if (for_iters[i] != index.as_var_ref()) {
        VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous for_iters[" << i << "] != index.as_var_ref(): " << for_iters[i] << " vs " << index.as_var_ref();
        return false;
      }
      // ir::Var iter_var = index.as_var_ref();
      // if (!iter_var2value.count(iter_var)) {
      //   VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous iter_var not found in iter_var2value: " << iter_var;
      //   return false;
      // }
      // ir::Expr iter_value = iter_var2value.at(iter_var);
      // if (!iter_value.as_var()) {
      //   VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous iter_value is not a var: " << iter_value;
      //   return false;
      // }
      // if (for_iters[i] != iter_value.as_var_ref()) {
      //   VLOG(6) << "YUHAN!! Inside CheckTensorIsContinuous for_iters[" << i << "] != iter_value.as_var_ref(): " << for_iters[i] << " vs " << iter_value.as_var_ref();
      //   return false;
      // }
      
    }

    // 检查最后一维的偏移表达式（如 j*C3 + k + C4）
    ir::Expr offset_expr = indices.back();
    std::vector<Var> replaced;
    std::vector<Expr> candidates;
    for (auto iter : iter_var2value) {
      replaced.push_back(iter.first);
      candidates.push_back(iter.second);
    }
    ReplaceExpr(&offset_expr, replaced, candidates);
    VLOG(6) << "YUHAN!!! IsLinearOffsetExpr simplified2 " << offset_expr;
    offset_expr = optim::ArithSimplify(offset_expr, IndexExpr::OptLevel::kLevel3);
    VLOG(6) << "YUHAN!!! IsLinearOffsetExpr simplified2 after ArithSimplify " << offset_expr;
    // 检查线性组合结构
    bool has_outer = false;    // 是否找到 outer_iter（j）
    bool has_inner = false;    // 是否找到 inner_iter（k）
    bool has_constant = false; // 是否找到常量项（C4）
    int outer_coeff = 1;       // outer_iter 的系数（C3）
    int constant_term = 0;     // 常量项的值（C4）
    if (!IsLinearOffsetExpr(offset_expr, for_iters[for_iters.size() - 2], for_iters.back(), iter_var2value,
                            &has_outer, &has_inner, &has_constant, &outer_coeff, &constant_term)) {
      VLOG(6) << "YUHAN!! Offset expr is not linear: " << offset_expr;
      return false;
    }

    // 条件2：C3 和 C4 必须是 2/4 的整数倍
    if (!IsAlignedConstant(outer_coeff, 2) && !IsAlignedConstant(outer_coeff) &&
        !IsAlignedConstant(constant_term, 2) && !IsAlignedConstant(constant_term) ) {
      VLOG(6) << "YUHAN!! Coefficient not aligned (required 2/4 multiple)";
      return false;
    }

    // 条件3：检查越界（假设已知张量形状为 shape[]）
    const ir::_Var_* var_ptr = for_iters.back().get();
    ir::Expr ub = var_ptr->upper_bound;
    ir::Expr lb = var_ptr->lower_bound;
    if (lb.get_constant() != 0 || outer_coeff != ub.get_constant()) {
      VLOG(6) << "YUHAN!! ub.get_constant() is" << ub.get_constant() << ", C3 is" << outer_coeff;
      return false;
    }
    // if (C3 * (/* C2 */) + (/* C3 */) >= inner_dim_size) {
    //   VLOG(6) << "YUHAN!! Offset access out of bounds";
    //   return false;
    // }
    return true;
  }
}

std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
CollectScheduleBlockTensorIndices(
    ir::Expr expr_schedule_block_realize,
    std::unordered_map<std::string, ir::Expr>* tensors) {
  std::unordered_map<std::string, std::vector<std::vector<ir::Expr>>>
      tensor_indices;
  ir::ir_utils::CollectIRNodesWithoutTensor(
      expr_schedule_block_realize,
      [&](const ir::Expr* expr) {
        if (expr->As<ir::Load>()) {
          auto* node = expr->As<ir::Load>();
          PADDLE_ENFORCE_NOT_NULL(
              node,
              ::common::errors::InvalidArgument(
                  "Expected Load node, but received nullptr."));
          auto* tensor = node->tensor.As<ir::_Tensor_>();
          PADDLE_ENFORCE_NOT_NULL(
              tensor,
              ::common::errors::InvalidArgument(
                  "Expected _Tensor_ node in load, but received nullptr."));
          tensor_indices[tensor->name].push_back(node->indices);
          tensors->insert({tensor->name, node->tensor});
          return true;
        }
        return false;
      },
      /* uniq_target = */ false);

  ir::ir_utils::CollectIRNodesWithoutTensor(
      expr_schedule_block_realize,
      [&](const ir::Expr* expr) {
        if (expr->As<ir::Store>()) {
          auto* node = expr->As<ir::Store>();
          PADDLE_ENFORCE_NOT_NULL(
              node,
              ::common::errors::InvalidArgument(
                  "Expected Load node, but received nullptr."));
          auto* tensor = node->tensor.As<ir::_Tensor_>();
          PADDLE_ENFORCE_NOT_NULL(
              tensor,
              ::common::errors::InvalidArgument(
                  "Expected _Tensor_ node in load, but received nullptr."));
          tensor_indices[tensor->name].push_back(node->indices);
          tensors->insert({tensor->name, node->tensor});
          return true;
        }
        return false;
      },
      /* uniq_target = */ false);
  return tensor_indices;
}

bool ScheduleBlockRealizeCanVectorize(
    ir::Expr expr_schedule_block_realize,
    const std::vector<ir::Var>& for_iters,
    const std::unordered_map<ir::Var, ir::Expr>& iter_var2value,
    const bool has_if_else_op,
    std::unordered_map<std::string, ir::Expr>* tensor_can_vectorize,
    std::unordered_map<std::string, std::vector<std::vector<bool>>>*
        broadcast_tensor_axis_info) {
  VLOG(6) << "YUHAN!!! Inside ScheduleBlockRealizeCanVectorize:";
  std::unordered_map<std::string, ir::Expr> tensors;
  auto tensor_indices =
      CollectScheduleBlockTensorIndices(expr_schedule_block_realize, &tensors);
  std::unordered_map<std::string, std::vector<std::vector<bool>>>
      broadcast_axis_info;
  for (auto& [tensor_name, indices_list] : tensor_indices) {
    VLOG(6) << "YUHAN!!! Checking Tensor " << tensor_name;
    for (auto& indices : indices_list) {
      if (CheckBroadcastTensorIsContinuous(
              tensor_name, indices, for_iters, iter_var2value)) {
        auto ba = CollectTensorBroadcastAxis(indices);
        VLOG(5) << "broadcast tensor name  " << tensor_name << "\n";
        broadcast_axis_info[tensor_name].emplace_back(ba);
        continue;
      }

      if (CheckTensorIsContinuous(indices, for_iters, iter_var2value)) {
        continue;
      }
      VLOG(6) << "YUHAN!!! Tensor " << tensor_name << " is not continuous";
      VLOG(6) << "YUHAN!!! Inside ScheduleBlockRealizeCanVectorize: return false";
      return false;
    }
  }

  if (!has_if_else_op) {
    tensor_can_vectorize->insert(tensors.begin(), tensors.end());
    broadcast_tensor_axis_info->insert(broadcast_axis_info.begin(),
                                       broadcast_axis_info.end());
  }

  return true;
}

void AnalysisGroupArgsWithVectorizeTensor(
    const std::unordered_set<std::string>& group_args,
    const std::unordered_map<std::string, ir::Expr>& vectorize_tensors,
    const std::unordered_map<std::string, std::vector<std::vector<bool>>>&
        tensor_broadcast_info,
    std::unordered_map<std::string, ir::Expr>* args_tensor_can_vectorize,
    std::unordered_map<std::string, std::vector<std::vector<bool>>>*
        args_tensor_deal_with_broadcast) {
  for (auto const& [tensor_name, tensor] : vectorize_tensors) {
    if (group_args.count(tensor_name)) {
      args_tensor_can_vectorize->insert({tensor_name, tensor});
    }

    if (group_args.count(tensor_name) &&
        tensor_broadcast_info.count(tensor_name)) {
      args_tensor_deal_with_broadcast->insert(
          {tensor_name, tensor_broadcast_info.at(tensor_name)});
    }
  }
  return;
}

}  // namespace

std::vector<int64_t> GetLoopStrides(const ir::Expr& body) {
  ir::Expr expr_block =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit)
          .GetSingle(body);
  auto* block = expr_block.As<ir::ScheduleBlockRealize>();
  auto& iter_values = block->iter_values;
  auto& iter_vars = block->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
  const std::vector<ir::Var> for_iters = GetAllForIters(body);

  const auto GetLoopIndex = [&](size_t var_index) {
    auto it = std::find(for_iters.begin(),
                        for_iters.end(),
                        iter_values[var_index].as_var_ref());
    PADDLE_ENFORCE_NE(it,
                      for_iters.end(),
                      ::common::errors::PreconditionNotMet(
                          "iter var %s was not found in loop vars: %s",
                          iter_values[var_index],
                          body));
    return std::distance(for_iters.begin(), it);
  };

  std::vector<ir::Expr> all_loads = GetRValueLoads(expr_block);
  std::vector<int64_t> loop_strides(for_iters.size());
  if (all_loads.empty()) {
    return loop_strides;
  }
  const ir::Expr largest_load = GetLargestLoad(all_loads);
  ir::Expr load_offset = largest_load.As<ir::Load>()->index();
  std::vector<int64_t> var_strides = GetVarStrides(load_offset, iter_vars);
  for (size_t i = 0; i < iter_vars.size(); i++) {
    loop_strides[GetLoopIndex(i)] = var_strides[i];
  }
  return loop_strides;
}

GroupVectorizeInfo GetGroupVectorizeInfo(
    const std::vector<ir::Expr>& op_compute_bodies,
    const std::unordered_set<std::string>& group_args) {
  VLOG(6) << "YUHAN!!! Inside GetGroupVectorizeInfo: ";
  bool can_vectorize = true;
  bool has_if_else_op = false;
  bool has_select_op = false;
  std::unordered_map<std::string, ir::Expr> vectorize_tensors;
  std::unordered_map<std::string, std::vector<std::vector<bool>>>
      tensor_broadcast_info;

  for (const auto& body : op_compute_bodies) {
    std::vector<ir::Expr> blocks =
        GetOpComputeBodyScheduleBlockRealizeExprSet(body);

    if (!ScheduleBlockRealizesShouldVectorizeCheck(blocks)) continue;
    ir::Expr expr_schedule_block_realize = blocks[0];
    const std::vector<ir::Var> for_iters =
        hlir::framework::pir::trivial_fusion_detail::GetAllForIters(body);
    std::unordered_map<ir::Var, ir::Expr> iter_var2value =
        ir::analyzer::GetIterVarToValueOfSBlock(expr_schedule_block_realize);
    bool current_block_has_if_else_op = false;
    if (ScheduleBlockRealizeHasSpecialOp(
            expr_schedule_block_realize,
            [](const ir::Expr* e) { return e->As<ir::IfThenElse>(); })) {
      current_block_has_if_else_op = true;
      has_if_else_op = true;
    }

    if (ScheduleBlockRealizeHasSpecialOp(
            expr_schedule_block_realize,
            [](const ir::Expr* e) { return e->As<ir::Select>(); })) {
      has_select_op = true;
    }

    if (ScheduleBlockRealizeCanVectorize(expr_schedule_block_realize,
                                         for_iters,
                                         iter_var2value,
                                         current_block_has_if_else_op,
                                         &vectorize_tensors,
                                         &tensor_broadcast_info))
      continue;
    VLOG(6) << "YUHAN!!! Inside GetGroupVectorizeInfo: can_vectorize = false";
    can_vectorize = false;
    break;
  }

  if (!can_vectorize) {
    return {false, has_if_else_op, has_select_op, {}, {}};
  }
  std::unordered_map<std::string, ir::Expr> args_tensor_can_vectorize;
  std::unordered_map<std::string, std::vector<std::vector<bool>>>
      args_tensor_broadcast_info;
  AnalysisGroupArgsWithVectorizeTensor(group_args,
                                       vectorize_tensors,
                                       tensor_broadcast_info,
                                       &args_tensor_can_vectorize,
                                       &args_tensor_broadcast_info);

  return {can_vectorize,
          has_if_else_op,
          has_select_op,
          std::move(args_tensor_can_vectorize),
          std::move(args_tensor_broadcast_info)};
}

}  // namespace ir
}  // namespace cinn
