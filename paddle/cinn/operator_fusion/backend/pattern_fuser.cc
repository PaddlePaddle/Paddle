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

#include "glog/logging.h"

#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/backend/pattern_fuser.h"

namespace cinn::fusion {

template <>
StmtPattern<BackendStage> ConvertToStmtPattern(
    const PatternContent<BackendStage>& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    PADDLE_ENFORCE_EQ(
        content.expr.has_value(),
        true,
        phi::errors::InvalidArgument(
            "The content.expr should have value in ConvertToStmtPattern."));
    return ReducePattern<BackendStage>({content.op},
                                       ReduceOp(content.expr.value()));
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    PADDLE_ENFORCE_EQ(
        content.expr.has_value(),
        true,
        phi::errors::InvalidArgument(
            "The content.expr should have value in ConvertToStmtPattern."));
    return TrivialPattern<BackendStage>(
        {content.op}, content.op, TrivialOp(content.expr.value()));
  } else {
    PADDLE_ENFORCE_EQ(
        false,
        true,
        phi::errors::InvalidArgument(
            "The content.op should be one of the following kinds: "
            "kReduction, kElementWise, kBroadcast, kInjective."));
    return UnsupportPattern<BackendStage>({content.op});
  }
}

// template StmtPattern<BackendStage> RT_x_RT(const
// ReduceTreePattern<BackendStage>& upstream, const
// ReduceTreePattern<BackendStage>& downstream);

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const ReduceTreePattern<BackendStage>& first,
    const TrivialPattern<BackendStage>& second) {
  return ReduceTreePlusTrivialPattern<BackendStage>(first, second);
}

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const TrivialPattern<BackendStage>& first,
    const ReducePattern<BackendStage>& second) {
  const auto& ops = UniqueConcatVector(GetOpsInPattern<BackendStage>(first),
                                       GetOpsInPattern<BackendStage>(second));
  const auto& reduce_op =
      cinn::hlir::framework::pir::trivial_fusion_detail::TrivalxOther_Fusion(
          first.trivial_op, second.reduce_op);
  return ReducePattern<BackendStage>(ops, reduce_op);
}

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const TrivialPattern<BackendStage>& first,
    const TrivialPattern<BackendStage>& second) {
  const auto& ops = UniqueConcatVector(GetOpsInPattern<BackendStage>(first),
                                       GetOpsInPattern<BackendStage>(second));
  const auto& trivial_op =
      cinn::hlir::framework::pir::trivial_fusion_detail::TrivalxOther_Fusion(
          first.trivial_op, second.trivial_op);
  return TrivialPattern<BackendStage>(ops, second.sink(), trivial_op);
}

/// Start: Tmp Transform Operation for ReduceTree
std::vector<FusionOp> ReduceTransformRecursive(
    ReduceOp reduce_op,
    const ReduceTreePattern<BackendStage>& reduce_tree_pattern,
    const std::vector<size_t>& fake_reduce_iter_idx = {}) {
  FusionOp root_op = reduce_op;
  VLOG(4) << "ReduceTransformRecursive: " << *_GetFuncBodyPointer(root_op);
  std::vector<FusionOp> result;
  for (const auto& child_tree : reduce_tree_pattern.childs()) {
    const auto& child_reduce_op = child_tree.GetRootPattern().reduce_op;
    auto transformed_nodes = cinn::hlir::framework::pir::trivial_fusion_detail::
        TransformReduceLoopRange(
            child_reduce_op, &root_op, fake_reduce_iter_idx);
    for (auto& node : transformed_nodes) {
      auto child_flatten =
          ReduceTransformRecursive(std::get<ReduceOp>(node), child_tree);
      result.insert(result.end(), child_flatten.begin(), child_flatten.end());
    }
  }
  result.push_back(root_op);
  VLOG(4) << "ReduceTransformRecursive: End";
  return result;
}

std::vector<FusionOp> ReduceTreeTrivialTransformRecursive(
    TrivialOp trivial_op,
    const ReduceTreePlusTrivialPattern<BackendStage>& rt_pattern) {
  FusionOp root_op = trivial_op;
  VLOG(4) << "ReduceTrivialTransformRecursive: "
          << *_GetFuncBodyPointer(root_op);
  std::vector<FusionOp> result;
  // for (const auto& child_tree : ) {
  //
  const auto& child_tree = rt_pattern.tree;
  const auto& child_reduce_op = child_tree.GetRootPattern().reduce_op;
  auto transformed_nodes = cinn::hlir::framework::pir::trivial_fusion_detail::
      TransformReduceLoopRange(
          child_reduce_op, &root_op, rt_pattern.fake_reduce_iter_idx);
  for (auto& node : transformed_nodes) {
    auto child_flatten = ReduceTransformRecursive(
        std::get<ReduceOp>(node), child_tree, rt_pattern.fake_reduce_iter_idx);
    result.insert(result.end(), child_flatten.begin(), child_flatten.end());
  }
  //}
  result.push_back(
      cinn::hlir::framework::pir::trivial_fusion_detail::SinkTrivialLoopAlign(
          std::get<TrivialOp>(root_op),
          rt_pattern.tree.GetRootPattern().reduce_op,
          rt_pattern.fake_reduce_iter_idx));
  VLOG(4) << "ReduceTrivialTransformRecursive End;";
  return result;
}

/// End: Tmp Transform Operation for reduce tree
///
struct FusionOp2Expr {
  std::vector<ir::Expr> operator()(const TrivialOp& op) {
    return {op.GetFuncBody()};
  }
  std::vector<ir::Expr> operator()(const ReduceOp& op) {
    const auto& t_r = SplitReduceOp(op);
    return {t_r.first.GetFuncBody(), t_r.second.GetFuncBody()};
  }
};

std::vector<ir::Expr> GetExprFromPattern(
    const StmtPattern<BackendStage>& pattern);

ir::Expr UnSqueezeExpr(const ir::Expr& expr,
                       const std::vector<int> padding_vec) {
  using cinn::hlir::framework::pir::trivial_fusion_detail::AppendBound;
  using cinn::hlir::framework::pir::trivial_fusion_detail::GetAllForIters;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildFors;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildRootScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      IsForIterVar;
  using cinn::hlir::framework::pir::trivial_fusion_detail::
      ExprTransformerUtils::ReplaceVarTransformer;
  using cinn::hlir::framework::pir::trivial_fusion_detail::
      ExprTransformerUtils::UnsqueezeForTransformer;
  VLOG(4) << "UnSqueezeExpr: " << expr
          << "\npadding vector: " << utils::Join(padding_vec, ", ");
  const auto& vars_in_expr = AppendBound(GetAllForIters(expr), expr);
  // get the all vars.
  int counter = 0;
  auto GenNextName = [&counter]() {
    counter += 1;
    return "expand_var_" + std::to_string(counter);
  };
  std::vector<ir::Var> vars;
  int pointer = 0;
  for (int i = 0; i < vars_in_expr.size() + padding_vec.size(); i++) {
    if (std::find(padding_vec.begin(), padding_vec.end(), i) !=
        padding_vec.end()) {
      vars.emplace_back(Expr(0), Expr(1), GenNextName());
    } else {
      vars.push_back(vars_in_expr[pointer++]);
    }
  }
  // update the is_reduce of expand_var.
  for (int i : padding_vec) {
    if (i == 0) {
      vars[i]->is_reduce_axis = false;
    } else {
      vars[i]->is_reduce_axis = vars[i - 1]->is_reduce_axis;
    }
  }

  // sequencely unsqueeze the ir::Expr.
  ir::Expr result = expr;
  for (int i : padding_vec) {
    if (i > 0) {
      result = UnsqueezeForTransformer((ChildFors * IsForIterVar(vars[i - 1])),
                                       vars[i])(result);
    } else {
      result = UnsqueezeForTransformer(ChildRootScheduleBlockRealizes,
                                       vars[i])(result);
    }
  }
  return result;
}

struct IrExprGetter {
  std::vector<ir::Expr> operator()(
      const TrivialPattern<BackendStage>& pattern) {
    return FusionOp2Expr()(pattern.trivial_op);
  }

  std::vector<ir::Expr> operator()(const ReducePattern<BackendStage>& pattern) {
    return FusionOp2Expr()(pattern.reduce_op);
  }

  std::vector<ir::Expr> operator()(
      const ReduceTreePattern<BackendStage>& pattern) {
    const auto& fusion_op =
        ReduceTransformRecursive(pattern.GetRootPattern().reduce_op, pattern);
    std::function<std::vector<ir::Expr>(const FusionOp& f)> func =
        [](const FusionOp& op) { return std::visit(FusionOp2Expr(), op); };
    return VectorFlatMap(fusion_op, func);
  }

  std::vector<ir::Expr> operator()(
      const ReduceTreePlusTrivialPattern<BackendStage>& pattern) {
    std::function<std::vector<ir::Expr>(const FusionOp& f)> func =
        [](const FusionOp& op) { return std::visit(FusionOp2Expr(), op); };
    const auto& fusion_ops = ReduceTreeTrivialTransformRecursive(
        pattern.sink_trivial.trivial_op, pattern);
    return VectorFlatMap(fusion_ops, func);
  }

  std::vector<ir::Expr> operator()(
      const HorizontalFusionPattern<BackendStage>& pattern) {
    std::vector<ir::Expr> result;
    VLOG(4) << "Get Fusion Ops from HorizontalFusionPattern: "
            << pattern.padding_patterns_.size();
    for (const auto& sub_pattern : pattern.padding_patterns_) {
      std::function<ir::Expr(ir::Expr)> func =
          [&sub_pattern](const ir::Expr& expr) {
            return UnSqueezeExpr(expr, sub_pattern.padding_pos);
          };
      result = ConcatVector(
          result, MapVector(GetExprFromPattern(sub_pattern.pattern), func));
    }
    return result;
  }

  std::vector<ir::Expr> operator()(
      const UnsupportPattern<BackendStage>& pattern) {
    PADDLE_ENFORCE_EQ(
        false,
        true,
        phi::errors::InvalidArgument(
            "The pattern is unsupported which leads to an error."));
  }
};

// tmp transform for reduce_tree and reduce_tree_trivial.
std::vector<ir::Tensor> GetOutputTensors(const ir::Expr& op_expr) {
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildTensorStores;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ScheduleBlockRealizeIsNotInit;
  const auto& tensors =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       ChildTensorStores)(op_expr);
  std::function<ir::Tensor(ir::Expr)> func = [](const ir::Expr& expr) {
    return expr.As<ir::Store>()->tensor.as_tensor_ref();
  };
  return MapVector(tensors, func);
}

std::vector<ir::Tensor> GetInputTensors(const ir::Expr& op_expr) {
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildTensorLoads;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ScheduleBlockRealizeIsNotInit;
  const auto& exprs =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       ChildTensorLoads)(op_expr);
  std::function<ir::Tensor(ir::Expr)> func = [](const ir::Expr& expr) {
    return expr.As<ir::Load>()->tensor.as_tensor_ref();
  };
  const auto& inputs = MapVector(exprs, func);
  const auto& outputs = GetOutputTensors(op_expr);
  return FilterVector(inputs, [&outputs](const ir::Tensor& tensor) {
    return std::find(outputs.begin(), outputs.end(), tensor) == outputs.end();
  });
}

std::vector<ir::Expr> TopoSort(const std::vector<ir::Expr>& op_exprs) {
  // Topo Sort is important for CINN GroupSchedule.
  std::map<ir::Tensor, std::vector<const ir::Expr*>> tensor2defining_op;
  std::map<ir::Tensor, std::vector<const ir::Expr*>> tensor2used_op;
  for (const auto& op : op_exprs) {
    auto inputs = GetInputTensors(op);
    auto outputs = GetOutputTensors(op);
    if (VLOG_IS_ON(4)) {
      VLOG(4) << "Ir::Expr is: \n" << op;
      VLOG(4) << "Inputs: ";
      for (const auto& input : inputs) {
        VLOG(4) << input->name;
      }
      VLOG(4) << "Outputs: ";
      for (const auto& output : outputs) {
        VLOG(4) << output->name;
      }
    }
    for (const auto& input : inputs) {
      tensor2used_op[input].push_back(&op);
    }
    for (const auto& output : outputs) {
      tensor2defining_op[output].push_back(&op);
    }
  }

  // Collect Downstreams
  std::map<const ir::Expr*, std::vector<const ir::Expr*>> op2downstreams;
  std::map<const ir::Expr*, int> degrees;
  for (const auto& op : op_exprs) {
    degrees[&op] = 0;
  }
  for (const auto& op : op_exprs) {
    auto outputs = GetOutputTensors(op);
    std::vector<const ir::Expr*> downstreams;
    for (const auto& output : outputs) {
      downstreams = ConcatVector(downstreams, tensor2used_op[output]);
    }
    for (const auto& downstream : downstreams) {
      degrees[downstream]++;
    }
    op2downstreams[&op] = downstreams;
  }

  // Topo Sort
  std::vector<const ir::Expr*> result;
  std::queue<const ir::Expr*> q;
  for (const auto& op : op_exprs) {
    if (degrees[&op] == 0) {
      q.push(&op);
    }
  }
  while (!q.empty()) {
    auto* cur = q.front();
    VLOG(4) << "Topo Sort Visit Order is:" << GetOutputTensors(*cur)[0]->name;
    q.pop();
    result.push_back(cur);
    for (const auto& downstream : op2downstreams[cur]) {
      degrees[downstream]--;
      if (degrees[downstream] == 0) {
        q.push(downstream);
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      result.size(),
      op_exprs.size(),
      phi::errors::InvalidArgument(
          "Required result.size() should be equal to op_exprs.size(). "));
  std::vector<ir::Expr> sorted_result;
  for (const auto& op : result) {
    sorted_result.push_back(*op);
  }
  return sorted_result;
}

std::vector<ir::Expr> GetExprFromPattern(
    const StmtPattern<BackendStage>& pattern) {
  const auto& results = std::visit(IrExprGetter(), pattern.variant());
  return TopoSort(results);
}

}  // namespace cinn::fusion
