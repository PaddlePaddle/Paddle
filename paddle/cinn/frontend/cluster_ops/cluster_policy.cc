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

#include "paddle/cinn/frontend/cluster_ops/cluster_policy.h"

namespace cinn::frontend::cluster_ops {

class LoopAlignableClusteringPolicy final : public ClusteringPolicy {
 public:
  explicit LoopAlignableClusteringPolicy(
      const pir::ShapeConstraintIRAnalysis* shape_analysis)
      : shape_analysis_(shape_analysis) {}

  bool CanActAsSink(const ShardableAxes4ValueT& ShardableAxes4Value,
                    const api::StmtPattern<FrontendPattern>& stmt) override {
    return IsSinkOpOutputFullyShardable(ShardableAxes4Value, stmt);
  }

  bool IsEdgeFusible(const ShardableAxes4ValueT& ShardableAxes4Value,
                     const api::StmtPattern<FrontendPattern>& src,
                     const api::StmtPattern<FrontendPattern>& dst) override {
    if (!IsSinkOpOutputFullyShardable(ShardableAxes4Value, src)) return false;
    if (!IsSinkOpOutputFullyShardable(ShardableAxes4Value, dst)) return false;
    if (!ReduceOpsSameShardable(ShardableAxes4Value, src, dst)) return false;
    if (!IsTotalLoopSizeEqual(src, dst)) return false;
    return true;
  }

  ClusteringResult MakeClusteringResult(
      const std::vector<StmtPatternPtrs>& stmts_list) {
    std::vector<LoopAlignableStmtsPattern> loop_alignable_list;
    for (const auto& stmt_ptrs : stmts_list) {
      loop_alignable_list.emplace_back(
          MakeLoopAlignableStmtsPattern(stmt_ptrs));
    }
    return ClusteringResult{
        .loop_alignable_list = std::move(loop_alignable_list),
    };
  }

 private:
  LoopAlignableStmtsPattern MakeLoopAlignableStmtsPattern(
      const std::vector<const StmtPattern*>& stmt_ptrs) {
    LoopAlignableStmtsPattern loop_alignable;
    loop_alignable.stmts.reserve(stmt_ptrs.size());
    for (const StmtPattern* stmt : stmt_ptrs) {
      loop_alignable.stmts.push_back(*stmt);
    }
    return loop_alignable;
  }

  bool IsTotalLoopSizeEqual(const StmtPattern& src, const StmtPattern& dst) {
    pir::Value src_value = GetStmtBigestShapeValue(src);
    pir::Value dst_value = GetStmtBigestShapeValue(dst);
    return shape_analysis_->IsProductEqual(
        src_value, 0, GetRank(src_value), dst_value, 0, GetRank(dst_value));
  }

  bool ReduceOpsSameShardable(const ShardableAxes4ValueT& ShardableAxes4Value,
                              const StmtPattern& src,
                              const StmtPattern& dst) {
    return std::visit(
        [&](const auto& src_impl, const auto& dst_impl) {
          return ReduceOpsSameShardableImpl(
              ShardableAxes4Value, src_impl, dst_impl);
        },
        src,
        dst);
  }

  template <typename SrcPatternT, typename DstPatternT>
  bool ReduceOpsSameShardableImpl(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const SrcPatternT& src,
      const DstPatternT& dst) {
    LOG(FATAL) << "Unimplemented. src_type: " << typeid(SrcPatternT).name()
               << ", dst_type: " << typeid(DstPatternT).name();
  }

  bool ReduceOpsSameShardableImpl(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const R& src,
      const PS& dst) {
    const auto* sink_op = src.reduce_op_pattern.reduce_op;
    pir::Value value =
        sink_op->result(GetOutputShardableAxesResultIdx(sink_op));
    const auto& shardable_axes = ShardableAxes4Value(value);
    CHECK(shardable_axes.has_value());
    return IsStmtSinkOpOutputFullyShardableImpl(src, *shardable_axes.value());
  }

  bool ReduceOpsSameShardableImpl(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const R& src,
      const R& dst) {
    const auto GetSoleOutputValue = [&](const R& reduce_pattern) {
      const auto* sink_op = src.reduce_op_pattern.reduce_op;
      pir::Value value =
          sink_op->result(GetOutputShardableAxesResultIdx(sink_op));
      return value;
    };
    const auto GetShardableAxes = [&](const R& reduce_pattern) {
      pir::Value value = GetSoleOutputValue(reduce_pattern);
      const auto& shardable_axes = ShardableAxes4Value(value);
      CHECK(shardable_axes.has_value());
      return shardable_axes.value();
    };
    const auto GetShardableAxesNames = [&](const R& reduce_pattern) {
      std::set<std::string> axis_names;
      for (const auto& shardable_axis : *GetShardableAxes(reduce_pattern)) {
        axis_names.insert(shardable_axis.axis_name);
      }
      return axis_names;
    };
    struct ShardibleAxisPair {
      std::optional<int> src_axis;
      std::optional<int> dst_axis;
    };
    const auto GetMatchedAxisPairs = [&]() {
      std::unordered_map<std::string, ShardibleAxisPair> matched_axis_pairs;
      for (const auto& src_sa : *GetShardableAxes(src)) {
        matched_axis_pairs[src_sa.axis_name].src_axis = src_sa.axis;
      }
      for (const auto& dst_sa : *GetShardableAxes(dst)) {
        matched_axis_pairs[dst_sa.axis_name].dst_axis = dst_sa.axis;
      }
      return matched_axis_pairs;
    };
    bool same_shardibility =
        (GetShardableAxesNames(src) == GetShardableAxesNames(dst));
    if (same_shardibility) {
      for (const auto& [axis_name, axis_pair] : GetMatchedAxisPairs()) {
        const auto& [src_axis, dst_axis] = axis_pair;
        CHECK(src_axis.has_value());
        CHECK(dst_axis.has_value());
        pir::Value src_value = GetSoleOutputValue(src);
        pir::Value dst_value = GetSoleOutputValue(dst);
        CHECK(shape_analysis_->IsProductEqual(
            src_value, {src_axis.value()}, dst_value, {dst_axis.value()}));
      }
    }
    return same_shardibility;
  }

  bool IsSinkOpOutputFullyShardable(
      const ShardableAxes4ValueT& ShardableAxes4Value,
      const StmtPattern& stmt) {
    const auto* sink_op = GetStmtSoleSinkOp(stmt);
    pir::Value value =
        sink_op->result(GetOutputShardableAxesResultIdx(sink_op));
    const auto& shardable_axes = ShardableAxes4Value(value);
    CHECK(shardable_axes.has_value());
    return IsStmtSinkOpOutputFullyShardable(stmt, *shardable_axes.value());
  }

  bool IsStmtSinkOpOutputFullyShardable(const StmtPattern& stmt,
                                        const ShardableAxes& shardable_axes) {
    return std::visit(
        [&](const auto& impl) {
          return IsStmtSinkOpOutputFullyShardableImpl(impl, shardable_axes);
        },
        stmt);
  }

  bool IsStmtSinkOpOutputFullyShardableImpl(
      const IS& injective_source, const ShardableAxes& shardable_axes) {
    return true;
  }

  bool IsStmtSinkOpOutputFullyShardableImpl(
      const PS& partial_shardable, const ShardableAxes& shardable_axes) {
    return true;
  }

  bool IsStmtSinkOpOutputFullyShardableImpl(
      const R& reduce_pattern, const ShardableAxes& shardable_axes) {
    const auto* reduce_op = reduce_pattern.reduce_op_pattern.reduce_op;
    if (reduce_op->isa<cinn::dialect::ReduceSumOp>()) {
      return IsCinnReduceSumOpOutputFullyShardable(reduce_op, shardable_axes);
    }
    LOG(FATAL) << "TODO(xiongkun). reduce_op name: " << reduce_op->name();
  }

  bool IsCinnReduceSumOpOutputFullyShardable(
      const pir::Operation* reduce_op, const ShardableAxes& shardable_axes) {
    const size_t input_rank = GetRank(reduce_op->operand_source(0));
    const auto& reduce_axes = GetReduceAxes(reduce_op);

    // no shardability if input reduced into one element.
    if (reduce_axes.empty()) return false;

    const auto& IsReduceAxis = [&](int axis) {
      return std::find(reduce_axes.begin(), reduce_axes.end(), axis) !=
             reduce_axes.end();
    };
    const auto& IsAxisSharded = [&](int axis) {
      const auto& Condition = [&](const auto& shardable_axis) {
        return shardable_axis.axis == axis;
      };
      return std::find_if(shardable_axes.begin(),
                          shardable_axes.end(),
                          Condition) != shardable_axes.end();
    };
    const bool keepdims = GetReduceOpKeepDims(reduce_op);
    if (keepdims) {
      const size_t output_rank = input_rank;
      CHECK(!reduce_axes.empty());
      for (int axis = 0; axis < output_rank; ++axis) {
        if (IsReduceAxis(axis)) continue;
        if (!IsAxisSharded(axis)) return false;
      }
      return true;
    } else {
      const int result_idx = GetOutputShardableAxesResultIdx(reduce_op);
      return GetRank(reduce_op->result(result_idx)) == shardable_axes.size();
    }
  }
  const pir::ShapeConstraintIRAnalysis* shape_analysis_;
};

std::shared_ptr<ClusteringPolicy> MakeLoopAlignableClusteringPolicy(
    const pir::ShapeConstraintIRAnalysis* shape_analysis) {
  return std::make_shared<LoopAlignableClusteringPolicy>(shape_analysis);
}
} // namespace cinn::frontend::cluster_ops
