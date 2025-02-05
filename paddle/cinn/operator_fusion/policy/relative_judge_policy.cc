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

#include "paddle/cinn/operator_fusion/policy/relative_judge_policy.h"
#include "paddle/cinn/operator_fusion/pattern.h"

namespace cinn::fusion {

bool RelativeJudgePolicy::IsDownstreamStmtDependReduceOp(
    pir::Operation* reduce, const StmtPattern& downstream) {
  const auto& values = GetPatternInputValues(downstream);
  for (const auto& value : reduce->results()) {
    if (std::find(values.begin(), values.end(), value) != values.end()) {
      return true;
    }
  }
  return false;
}

std::optional<ReducePattern> RelativeJudgePolicy::GetDownstreamFromCandidate(
    const ReducePattern& upstream,
    const std::vector<ReducePattern>& candidates) {
  pir::Operation* reduce = upstream.GetReduceOp();
  for (const auto& candidate : candidates) {
    if (IsDownstreamStmtDependReduceOp(reduce, candidate)) {
      return candidate;
    }
  }
  return {};
}

std::pair<std::vector<DimUsage>, std::vector<DimUsage>> SplitReduceDims(
    const ShardableAxesSignature& signature, pir::Operation* op) {
  const auto& v = op->operand_source(0);
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> output_names_set(output_names.begin(),
                                         output_names.end());

  std::vector<DimUsage> reduce_dims;
  std::vector<DimUsage> non_reduce_dims;
  auto usage_idx = GetUsageIdx(v, op);

  int idx = 0;
  for (const auto& in : input_names) {
    if (output_names_set.count(in) == 0) {
      reduce_dims.emplace_back(v, idx, usage_idx);
    } else {
      non_reduce_dims.emplace_back(v, idx, usage_idx);
    }
    idx += 1;
  }

  if (VLOG_IS_ON(4)) {
    std::stringstream ss;
    ss << "SplitReduceDims:\nreduce_dims:\n";
    for (const auto& dim : reduce_dims) {
      ss << dim.DebugStr() << "\n";
    }
    ss << "non_reduce_dims:\n";
    for (const auto& dim : non_reduce_dims) {
      ss << dim.DebugStr() << "\n";
    }
    VLOG(4) << ss.str();
  }

  return {reduce_dims, non_reduce_dims};
}

std::pair<std::vector<DimUsage>, std::vector<DimUsage>>
RelativeJudgePolicy::SplitFirstIfRelatedBySecond(
    const std::vector<DimUsage>& targets,
    const std::vector<DimUsage>& related_with) {
  std::vector<DimUsage> related_dims;
  std::vector<DimUsage> non_related_dims;

  bool is_related = false;
  for (auto& target_dim : targets) {
    is_related = false;
    for (auto& related_dim : related_with) {
      if (IsRelated(related_dim, target_dim)) is_related = true;
    }
    if (is_related) {
      related_dims.push_back(target_dim);
    } else {
      non_related_dims.push_back(target_dim);
    }
  }

  if (VLOG_IS_ON(4)) {
    std::stringstream ss;
    ss << "SplitFirstIfRelatedBySecond:\nrelated_dims:\n";
    for (const auto& dim : related_dims) {
      ss << dim.DebugStr() << "\n";
    }
    ss << "non_related_dims:\n";
    for (const auto& dim : non_related_dims) {
      ss << dim.DebugStr() << "\n";
    }
    VLOG(4) << ss.str();
  }

  return {related_dims, non_related_dims};
}

bool ElementwiseEqual(const std::vector<DimUsage>& first,
                      const std::vector<DimUsage>& second) {
  const auto GetDimInfo = [](const std::vector<DimUsage>& dims)
      -> std::unordered_map<symbol::DimExpr, int> {
    std::unordered_map<symbol::DimExpr, int> result;
    for (const auto& dim : dims) {
      VLOG(4) << "dim: " << dim.DebugStr();
      symbol::DimExpr value = dim.GetSymbolicDim();
      VLOG(4) << "value: " << value;
      if (result.find(value) == result.end()) {
        result[value] = 1;
      } else {
        result[value] += 1;
      }
    }
    return result;
  };

  const std::unordered_map<symbol::DimExpr, int>& first_dims =
      GetDimInfo(first);
  const std::unordered_map<symbol::DimExpr, int>& second_dims =
      GetDimInfo(second);

  if (first_dims.size() != second_dims.size()) return false;
  for (const auto& [dim_value, count] : first_dims) {
    if (second_dims.find(dim_value) == second_dims.end() ||
        second_dims.at(dim_value) != count)
      return false;
  }
  return true;
}

symbol::DimExpr GetProductDimExprForValueDims(
    const std::vector<DimUsage>& dims) {
  if (dims.empty()) {
    return 0;
  }
  std::vector<int> dim_idx;
  for (const auto& dim : dims) {
    dim_idx.emplace_back(dim.idx_);
  }
  return dims[0].shape_analysis().GetProductDimExpr(dims[0].v_, dim_idx);
}

bool IsProductSmallerOrEqual(const std::vector<DimUsage>& first,
                             const std::vector<DimUsage>& second) {
  if (first.empty()) return true;
  const auto& first_product = GetProductDimExprForValueDims(first);
  const auto& second_product = GetProductDimExprForValueDims(second);
  const auto& shape_analysis = first[0].shape_analysis();
  if (second_product.isa<int64_t>() && first_product.isa<int64_t>()) {
    VLOG(4) << "Static Shape: left is "
            << std::get<int64_t>(first_product.variant()) << " ; right is "
            << std::get<int64_t>(second_product.variant());
    return std::get<int64_t>(first_product.variant()) <=
           std::get<int64_t>(second_product.variant());
  }
  return shape_analysis.IsEqual(first_product, second_product);
}

bool RelativeJudgePolicy::ReduceTreeGrownCanMerge(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  const auto& upstream_tree =
      std::get<ReduceTreePattern>(upstream->stmt_pattern());
  const auto& downstream_tree =
      std::get<ReduceTreePattern>(downstream->stmt_pattern());

  VLOG(4) << "upstream: \n" << OpsDebugStr(GetOpsInPattern(upstream_tree));
  VLOG(4) << "upstream->children_num: " << upstream_tree.children().size();
  VLOG(4) << "downstream: \n" << OpsDebugStr(GetOpsInPattern(downstream_tree));
  VLOG(4) << "downstream->children_num: " << downstream_tree.children().size();

  const auto& maybe_downstream_op = GetDownstreamFromCandidate(
      upstream_tree.GetRootPattern(), downstream_tree.FlattenReducePattern());
  int idx = 0;
  for (const auto& r_pattern : downstream_tree.children()) {
    idx += 1;
    VLOG(4) << "downstream_tree.reduce_patterns_"
            << "[" << idx << "]" << OpsDebugStr(GetOpsInPattern(r_pattern));
  }
  if (!maybe_downstream_op.has_value()) {
    VLOG(4) << "can't find candidate from patterns. can fuse return false.";
    return false;
  }
  const pir::Value& reduce_out_value =
      upstream_tree.GetRootPattern().GetReduceOp()->result(0);
  auto downstream_connect_ops =
      FindUserOp(downstream_tree.ops(), reduce_out_value);
  pir::Operation* downstream_reduce_op =
      maybe_downstream_op.value().GetReduceOp();

  const auto& [downstream_reduce_dims, downstream_non_reduce_dims] =
      SplitReduceDims(axes_info_.GetSignature(downstream_reduce_op),
                      downstream_reduce_op);

  std::vector<DimUsage> upstream_output_dims;
  for (const auto& op : downstream_connect_ops) {
    auto dim_usages =
        GetValueUsage(reduce_out_value, GetUsageIdx(reduce_out_value, op));
    upstream_output_dims.insert(
        upstream_output_dims.end(), dim_usages.begin(), dim_usages.end());
  }
  const auto& [related, _UNUSED] =
      SplitFirstIfRelatedBySecond(downstream_reduce_dims, upstream_output_dims);
  if (related.size() > 0) {
    return false;
  }

  auto upstream_reduce_op = upstream_tree.GetRootPattern().GetReduceOp();
  const auto& [upstream_reduce_dims, _unused_dims] = SplitReduceDims(
      axes_info_.GetSignature(upstream_reduce_op), upstream_reduce_op);
  if (upstream_reduce_dims.size() != downstream_reduce_dims.size()) {
    return false;
  }
  for (size_t i = 0; i < upstream_reduce_dims.size(); ++i) {
    if (upstream_reduce_dims[i].GetSymbolicDim() !=
        downstream_reduce_dims[i].GetSymbolicDim()) {
      return false;
    }
  }

  return true;
}

bool RelativeJudgePolicy::ReducePlusTrivialCanMerge(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  const auto& [upstream_reduce_dims, upstream_non_reduce_dims] =
      SplitReduceDims(axes_info_.GetSignature(upstream->sink_op()),
                      upstream->sink_op());

  // usage_idx is not important, for this is downstream output value
  // downstream output value must have been used for there is yield op, so
  // usage_idx==0 exists
  const auto& [_UNUSED, non_related_dims] = SplitFirstIfRelatedBySecond(
      GetValueUsage(downstream->sink_op()->result(0), 0),
      upstream_non_reduce_dims);

  const auto& fakes = GetFakeReduceIterIdx(upstream, downstream);
  const auto& downstream_free_dims = GatherVectorExcept(
      GetValueUsage(downstream->sink_op()->result(0), 0), fakes);

  // TODO(huangjiyi): support fusion when fake_reduce.size < reduce.size
  bool res = ElementwiseEqual(non_related_dims, upstream_reduce_dims) ||
             (IsProductSmallerOrEqual(downstream_free_dims,
                                      upstream_non_reduce_dims) &&
              (fakes.empty() || fakes.size() == upstream_reduce_dims.size()));

  if (res && downstream_free_dims.empty() && !fakes.empty()) {
    res = false;
  }

  VLOG(4) << "ReducePlusTrivialCanMerge: " << res;
  return res;
}

bool RelativeJudgePolicy::CanFuse(const PatternNodePtr& upstream,
                                  const PatternNodePtr& downstream) {
  if (std::holds_alternative<ReduceTreePattern>(upstream->stmt_pattern()) &&
      std::holds_alternative<TrivialPattern>(downstream->stmt_pattern())) {
    return ReducePlusTrivialCanMerge(upstream, downstream);
  }
  if (std::holds_alternative<ReduceTreePattern>(upstream->stmt_pattern()) &&
      std::holds_alternative<ReduceTreePattern>(downstream->stmt_pattern())) {
    return ReduceTreeGrownCanMerge(upstream, downstream);
  }
  return true;  // other case.
}

std::vector<size_t> RelativeJudgePolicy::GetFakeReduceIterIdx(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  if (!std::holds_alternative<ReduceTreePattern>(upstream->stmt_pattern()) &&
      !std::holds_alternative<TrivialPattern>(downstream->stmt_pattern())) {
    PADDLE_THROW("Illegal Call GetFakeReduceIterIdx");
  }

  const auto& [upstream_reduce_dims, upstream_non_reduce_dims] =
      SplitReduceDims(axes_info_.GetSignature(upstream->sink_op()),
                      upstream->sink_op());

  const auto& [_UNUSED, trivial_reorder_dims] = SplitFirstIfRelatedBySecond(
      GetValueUsage(downstream->sink_op()->result(0), 0),
      upstream_non_reduce_dims);

  std::unordered_set<DimUsage, DimUsageHash> visited_dims;
  std::vector<size_t> result;
  for (auto& reduce_dim : upstream_reduce_dims) {
    for (auto& trivial_dim : trivial_reorder_dims) {
      if (visited_dims.find(trivial_dim) == visited_dims.end() &&
          trivial_dim.SymbolicEqualTo(reduce_dim)) {
        visited_dims.emplace(trivial_dim);
        result.emplace_back(trivial_dim.idx_);
        break;
      }
    }
  }
  VLOG(4) << "FakeReduceIterIdx: " << cinn::utils::Join(result, ", ");
  return result;
}

}  // namespace cinn::fusion
