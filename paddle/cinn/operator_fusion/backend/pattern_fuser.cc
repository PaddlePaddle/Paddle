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
    CHECK(content.expr.has_value());
    return ReducePattern<BackendStage>({content.op},
                                       ReduceOp(content.expr.value()));
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    CHECK(content.expr.has_value());
    return TrivialPattern<BackendStage>(
        {content.op}, content.op, TrivialOp(content.expr.value()));
  } else {
    CHECK(false);
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

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const HorizontalFusionPattern<BackendStage>& first,
    const HorizontalFusionPattern<BackendStage>& second) {
  return HorizontalFusionPattern<BackendStage>({first, second});
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

std::vector<FusionOp> GetFusionOpFromPattern(
    const StmtPattern<BackendStage>& pattern);

struct FusionOpGetter {
  std::vector<FusionOp> operator()(
      const TrivialPattern<BackendStage>& pattern) {
    return {pattern.trivial_op};
  }

  std::vector<FusionOp> operator()(const ReducePattern<BackendStage>& pattern) {
    return {pattern.reduce_op};
  }

  std::vector<FusionOp> operator()(
      const ReduceTreePattern<BackendStage>& pattern) {
    return ReduceTransformRecursive(pattern.GetRootPattern().reduce_op,
                                    pattern);
  }

  std::vector<FusionOp> operator()(
      const ReduceTreePlusTrivialPattern<BackendStage>& pattern) {
    return ReduceTreeTrivialTransformRecursive(pattern.sink_trivial.trivial_op,
                                               pattern);
  }

  std::vector<FusionOp> operator()(
      const HorizontalFusionPattern<BackendStage>& pattern) {
    std::vector<FusionOp> result;
    VLOG(4) << "Get Fusion Ops from HorizontalFusionPattern: "
            << pattern.patterns_.size();
    for (const auto& sub_pattern : pattern.patterns_) {
      result = ConcatVector(result, GetFusionOpFromPattern(sub_pattern));
    }
    return result;
  }

  std::vector<FusionOp> operator()(
      const UnsupportPattern<BackendStage>& pattern) {
    CHECK(false) << "Not Implemented.";
  }
};

// tmp transform for reduce_tree and reduce_tree_trivial.
std::vector<FusionOp> GetFusionOpFromPattern(
    const StmtPattern<BackendStage>& pattern) {
  return std::visit(FusionOpGetter(), pattern.variant());
}

struct FusionOp2Expr {
  std::vector<ir::Expr> operator()(const TrivialOp& op) {
    return {op.GetFuncBody()};
  }
  std::vector<ir::Expr> operator()(const ReduceOp& op) {
    const auto& t_r = SplitReduceOp(op);
    return {t_r.first.GetFuncBody(), t_r.second.GetFuncBody()};
  }
};

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
    q.pop();
    result.push_back(cur);
    for (const auto& downstream : op2downstreams[cur]) {
      degrees[downstream]--;
      if (degrees[downstream] == 0) {
        q.push(downstream);
      }
    }
  }
  CHECK_EQ(result.size(), op_exprs.size());
  std::vector<ir::Expr> sorted_result;
  for (const auto& op : result) {
    sorted_result.push_back(*op);
  }
  return sorted_result;
}

std::vector<ir::Expr> GetExprFromPattern(
    const StmtPattern<BackendStage>& pattern) {
  const auto& fusion_ops = GetFusionOpFromPattern(pattern);
  VLOG(4) << "GetExprFromPattern from " << GetPatternName(pattern)
          << " : get fusion op number : " << fusion_ops.size();
  std::vector<ir::Expr> results;
  for (const auto& op : fusion_ops) {
    results = ConcatVector(results, std::visit(FusionOp2Expr(), op));
  }
  return TopoSort(results);
}

}  // namespace cinn::fusion
