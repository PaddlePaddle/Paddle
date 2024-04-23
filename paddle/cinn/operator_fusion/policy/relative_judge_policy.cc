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
#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"

namespace cinn::fusion {

template <typename T>
bool RelativeJudgePolicy<T>::IsDownstreamStmtDependReduceOp(
    pir::Operation* reduce, const StmtPattern<T>& downstream) {
  const auto& values = GetPatternInputValues(downstream);
  for (const auto& value : reduce->results()) {
    if (std::find(values.begin(), values.end(), value) != values.end()) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::optional<ReducePattern<T>>
RelativeJudgePolicy<T>::GetDownstreamFromCandidate(
    const ReducePattern<T>& upstream,
    const std::vector<ReducePattern<T>>& candidates) {
  pir::Operation* reduce = upstream.GetReduceOp();
  for (const auto& candidate : candidates) {
    if (IsDownstreamStmtDependReduceOp(reduce, candidate)) {
      return candidate;
    }
  }
  return {};
}

SplitDims SplitReduceInputDimsIfRelatedWithNonReduceAxis(
    const ShardableAxesSignature& signature, pir::Operation* op) {
  const auto& v = op->operand_source(0);
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> output_names_set(output_names.begin(),
                                         output_names.end());
  auto result = SplitDims();
  int idx = 0;
  for (const auto& in : input_names) {
    if (output_names_set.count(in) == 0) {
      result.non_related.emplace_back(v, idx);
    } else {
      result.related.emplace_back(v, idx);
    }
    idx += 1;
  }
  return result;
}

SplitDims SplitReduceOutputDimsIfRelatedWithNonReduceAxis(
    const ShardableAxesSignature& signature, const pir::Operation* op) {
  const auto& v = op->result(0);
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> input_names_set(input_names.begin(), input_names.end());
  auto result = SplitDims();
  int idx = 0;
  for (const auto& name : output_names) {
    if (input_names_set.count(name) == 0) {
      result.non_related.emplace_back(v, idx);
    } else {
      result.related.emplace_back(v, idx);
    }
    idx += 1;
  }
  return result;
}

template <typename T>
bool RelativeJudgePolicy<T>::IsBroadcastEdge(
    const std::vector<ValueDim>& upstream_out_dims,
    const std::vector<ValueDim>& downstream_reduce_dims) {
  VLOG(4) << "IsBroadcastEdge: upstream_out_dims.size()"
          << upstream_out_dims.size();
  VLOG(4) << "IsBroadcastEdge: downstream_reduce_dims.size()"
          << downstream_reduce_dims.size();

  for (const auto& downstream_reduce_dim : downstream_reduce_dims) {
    for (const auto& upstream_out_dim : upstream_out_dims) {
      VLOG(4) << "upstream_out_dim: " << upstream_out_dim.DebugStr()
              << " downstream_reduce_dim: " << downstream_reduce_dim.DebugStr();
      if (IsRelated(upstream_out_dim, downstream_reduce_dim)) {
        return false;
      }
    }
  }

  VLOG(4) << "IsBroadcastEdge";
  return true;
}

template <typename T>
bool RelativeJudgePolicy<T>::ReduceTreeGrownCanMerge(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  const auto& upstream_tree =
      std::get<ReduceTreePattern<T>>(upstream->stmt_pattern());
  VLOG(4) << "upstream->stmt_pattern():"
          << OpsDebugStr(GetOpsInPattern<T>(upstream_tree));
  const auto& downstream_tree =
      std::get<ReduceTreePattern<T>>(downstream->stmt_pattern());
  VLOG(4) << "downstream->stmt_pattern()"
          << OpsDebugStr(GetOpsInPattern<T>(downstream_tree));
  const auto& maybe_downstream_op = GetDownstreamFromCandidate(
      upstream_tree.GetRootPattern(), downstream_tree.FlattenReducePattern());
  int idx = 0;
  for (const auto& r_pattern : downstream_tree.childs()) {
    idx += 1;
    VLOG(4) << "downstream_tree.reduce_patterns_"
            << "[" << idx << "]" << OpsDebugStr(GetOpsInPattern<T>(r_pattern));
  }
  if (!maybe_downstream_op.has_value()) {
    VLOG(4) << "can't find candidate from patterns. can fuse return false.";
    return false;
  }
  const pir::Value& reduce_out_value =
      upstream_tree.GetRootPattern().GetReduceOp()->result(0);
  pir::Operation* downstream_reduce_op =
      maybe_downstream_op.value().GetReduceOp();
  const auto& split_reduce_dim_result =
      SplitReduceInputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(downstream_reduce_op), downstream_reduce_op);
  VLOG(4) << split_reduce_dim_result.DebugStr();
  const auto& upstream_output_dims = GetAllValueDimFromValue(reduce_out_value);
  auto res = IsBroadcastEdge(upstream_output_dims,
                             split_reduce_dim_result.non_related);
  VLOG(4) << "ReduceTreeGrownCanMerge: " << res;
  return res;
}

template <typename T>
SplitDims RelativeJudgePolicy<T>::SplitDimsWithRelationship(
    const std::vector<ValueDim>& targets,
    const std::vector<ValueDim>& related_with) {
  VLOG(4) << "SplitDimsWithRelationship";
  auto result = SplitDims();
  bool is_related = false;
  for (auto& target_dim : targets) {
    is_related = false;
    for (auto& related_dim : related_with) {
      if (IsRelated(related_dim, target_dim)) is_related = true;
    }
    if (is_related) {
      result.related.push_back(target_dim);
    } else {
      result.non_related.push_back(target_dim);
    }
  }

  return result;
}

bool DimsEqual(const std::vector<ValueDim>& first,
               const std::vector<ValueDim>& second) {
  const auto GetDimInfo = [](const std::vector<ValueDim>& dims)
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
  VLOG(4) << "GetDimInfo";
  const std::unordered_map<symbol::DimExpr, int>& first_dims =
      GetDimInfo(first);
  VLOG(4) << "GetDimInfo";
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

template <typename T>
std::vector<ValueDim> RelativeJudgePolicy<T>::getUpstreamReduceDims(
    const PatternNodePtr<T>& upstream,
    ShardableAxesInfoManager& axes_info) {  // NOLINT
  const auto& split_reduce_input_dims_result =
      SplitReduceInputDimsIfRelatedWithNonReduceAxis(
          axes_info.GetSignature(upstream->sink_op()), upstream->sink_op());
  return split_reduce_input_dims_result.non_related;
}

template <typename T>
std::vector<ValueDim> RelativeJudgePolicy<T>::getDownstreamUnrelatedDims(
    const PatternNodePtr<T>& upstream,
    const PatternNodePtr<T>& downstream,
    ShardableAxesInfoManager& axes_info) {  // NOLINT
  const auto& split_reduce_output_dims_result =
      SplitReduceOutputDimsIfRelatedWithNonReduceAxis(
          axes_info.GetSignature(upstream->sink_op()), upstream->sink_op());
  const auto& upstream_non_reduce_dims =
      split_reduce_output_dims_result.related;
  const auto& split_trivial_dims_result = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op()->result(0)),
      upstream_non_reduce_dims);
  VLOG(4) << split_trivial_dims_result.DebugStr();
  return split_trivial_dims_result.non_related;
}

template <typename T>
bool RelativeJudgePolicy<T>::ReducePlusTrivialCanMerge(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  VLOG(4) << "RT can fuse";
  const auto& upstream_reduce_dims =
      getUpstreamReduceDims(upstream, axes_info_);
  const auto& downstream_non_related_dims =
      getDownstreamUnrelatedDims(upstream, downstream, axes_info_);
  auto res = DimsEqual(downstream_non_related_dims, upstream_reduce_dims);
  res = res || IsFlattenDimSmaller(upstream, downstream);
  VLOG(4) << "ReducePlusTrivialCanMerge: " << res;
  return res;
}

namespace {

std::vector<ValueDim> GatherDimsExcept(const std::vector<ValueDim>& dims,
                                       const std::vector<size_t>& except) {
  std::vector<ValueDim> result;
  for (size_t i = 0; i < dims.size(); i++) {
    if (std::find(except.begin(), except.end(), i) == except.end()) {
      result.emplace_back(dims[i]);
    }
  }
  return result;
}

symbol::DimExpr GetProductDimExprForValueDims(
    const std::vector<ValueDim>& dims) {
  if (dims.empty()) {
    return 0;
  }
  std::vector<int> dim_idx;
  for (const auto& dim : dims) {
    dim_idx.emplace_back(dim.idx_);
  }
  return dims[0].shape_analysis().GetProductDimExpr(dims[0].v_, dim_idx);
}

bool IsProductSmallerOrEqual(const std::vector<ValueDim>& first,
                             const std::vector<ValueDim>& second) {
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

}  // namespace

template <typename T>
bool RelativeJudgePolicy<T>::IsFlattenDimSmaller(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  const auto& fakes = GetFakeReduceIterIdx(upstream, downstream);
  VLOG(4) << "IsFlattenDimSmaller: fake is " << utils::Join(fakes, ",");
  const auto& downstream_free_dims = GatherDimsExcept(
      GetAllValueDimFromValue(downstream->sink_op()->result(0)), fakes);
  const auto& upstream_free_dims =
      GetAllValueDimFromValue(upstream->sink_op()->result(0));

  bool res = IsProductSmallerOrEqual(downstream_free_dims, upstream_free_dims);
  VLOG(4) << "IsFlattenDimSmaller: " << res;
  return res;
}

template <typename T>
bool RelativeJudgePolicy<T>::CanFuse(const PatternNodePtr<T>& upstream,
                                     const PatternNodePtr<T>& downstream) {
  if (std::holds_alternative<ReduceTreePattern<T>>(upstream->stmt_pattern()) &&
      std::holds_alternative<TrivialPattern<T>>(downstream->stmt_pattern())) {
    return ReducePlusTrivialCanMerge(upstream, downstream);
  }
  if (std::holds_alternative<ReduceTreePattern<T>>(upstream->stmt_pattern()) &&
      std::holds_alternative<ReduceTreePattern<T>>(
          downstream->stmt_pattern())) {
    return ReduceTreeGrownCanMerge(upstream, downstream);
  }
  return true;  // other case.
}

template <typename T>
std::vector<size_t> RelativeJudgePolicy<T>::GetFakeReduceIterIdx(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  if (!std::holds_alternative<ReduceTreePattern<T>>(upstream->stmt_pattern()) &&
      !std::holds_alternative<TrivialPattern<T>>(downstream->stmt_pattern())) {
    PADDLE_THROW("Illegal Call GetFakeReduceIterIdx");
  }

  // TODO(xiongkun): replace after fix bug in relation that if has multi path in
  // graph const auto& split_reduce_dims_result =
  // SplitReduceInputDimsIfRelatedWithNonReduceAxis(
  // axes_info_.GetSignature(upstream->sink_op()), upstream->sink_op());

  // const auto& upstream_reduce_dims = split_reduce_dims_result.non_related;
  // const auto& upstream_non_reduce_dims = split_reduce_dims_result.related;
  //

  const auto& split_reduce_input_dims_result =
      SplitReduceInputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(upstream->sink_op()), upstream->sink_op());
  VLOG(4) << split_reduce_input_dims_result.DebugStr();
  const auto& upstream_reduce_dims = split_reduce_input_dims_result.non_related;
  const auto& split_reduce_output_dims_result =
      SplitReduceOutputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(upstream->sink_op()), upstream->sink_op());
  VLOG(4) << split_reduce_input_dims_result.DebugStr();
  const auto& upstream_non_reduce_dims =
      split_reduce_output_dims_result.related;

  // =======================

  const auto& split_trivial_dims_result = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op()->result(0)),
      upstream_non_reduce_dims);

  const auto& trivial_reorder_dims = split_trivial_dims_result.non_related;

  // CHECK(upstream_reduce_dims.size() == trivial_reorder_dims.size() ||
  // trivial_reorder_dims.size() == 0);
  std::unordered_set<ValueDim, ValueDimHash> visited_dims;
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

template class RelativeJudgePolicy<FrontendStage>;
template class RelativeJudgePolicy<BackendStage>;

}  // namespace cinn::fusion
