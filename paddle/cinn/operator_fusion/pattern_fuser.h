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

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/utils.h"

// This file is the protocol of the pattern fuser. Please implement
// ConvertToStmtPattern and MergePatternImpl in the specializations.

namespace cinn::fusion {

template <typename T>
ReducePattern<T> ToReducePattern(const StmtPattern<T>& second) {
  return std::get<ReducePattern<T>>(second);
}

template <typename T>
std::string GetPatternName(const StmtPattern<T>& s) {
  return std::visit([](const auto& impl) { return impl.name(); }, s.variant());
}

template <typename T>
StmtPattern<T> ConvertToStmtPattern(const PatternContent<T>& content);

template <typename T>
std::vector<pir::Operation*> GetOpsInPattern(const StmtPattern<T>& pattern) {
  return std::visit([](const auto& impl) { return impl.ops(); },
                    pattern.variant());
}

using LoopFramework = std::vector<symbol::DimExpr>;

// std::optional({}) means not sure.
// std::optional will cause SegmentFault, TODO: fix this.
using MaybeLoopFramework = LoopFramework;

template <typename T>
MaybeLoopFramework GetLoopFramework(const StmtPattern<T>& pattern);

template <typename T>
struct LoopFrameworkVisitor {
  MaybeLoopFramework operator()(const ReducePattern<T>& pattern) {
    pir::Operation* reduce_op = pattern.GetReduceOp();
    const auto& flatten_loops = GetDimExprsFromValue(reduce_op->result(0));
    const auto& reduce_axes = GetReduceAxisIdx(reduce_op);
    const auto& reduce_loops = GatherVector(
        GetDimExprsFromValue(reduce_op->operand(0).source()), reduce_axes);
    return ConcatVector(flatten_loops, reduce_loops);
  }

  MaybeLoopFramework operator()(const ReduceTreePattern<T>& pattern) {
    return GetLoopFramework(StmtPattern<T>(pattern.GetRootPattern()));
  }

  MaybeLoopFramework operator()(const TrivialPattern<T>& pattern) {
    pir::Operation* t_op = pattern.sink();
    const auto& dims = GetDimExprsFromValue(t_op->result(0));
    const auto& exprs = GetDimExprsFromValue(t_op->result(0));
    return exprs;
  }

  MaybeLoopFramework operator()(const HorizontalFusionPattern<T>& pattern) {
    // Horizontal Fusion must have the same loop framework.
    VLOG(4) << "Get horizontal fusion pattern for loop framework.";
    const auto& base_exprs = GetLoopFramework(
        StmtPattern<T>(pattern.padding_patterns_.back().pattern));
    const auto& padding_vector = pattern.padding_patterns_.back().padding_pos;
    std::vector<symbol::DimExpr> exprs(
        base_exprs.size() + padding_vector.size(), 1);
    int pointer = 0;
    for (int i = 0; i < exprs.size(); i++) {
      if (std::find(padding_vector.begin(), padding_vector.end(), i) ==
          padding_vector.end()) {
        exprs[i] = base_exprs[pointer++];
      }
    }
    return exprs;
  }

  MaybeLoopFramework operator()(
      const ReduceTreePlusTrivialPattern<T>& pattern) {
    const auto& sink_trivial = pattern.sink_trivial;
    const auto& trivial_loop =
        GetLoopFramework(StmtPattern<T>(pattern.sink_trivial));
    if (pattern.fake_reduce_iter_idx.empty()) {
      // we add reduce loop to the end;
      int reduce_axes_len =
          GetReduceAxisIdx(pattern.tree.GetRootPattern().GetReduceOp()).size();
      const auto& reduce_loop =
          GetLoopFramework(StmtPattern<T>(pattern.tree.GetRootPattern()));
      return ConcatVector(
          trivial_loop,
          SliceVector(reduce_loop, -reduce_axes_len, reduce_loop.size()));
    } else {
      // we always put fake into the end to make the loop framework consistent.
      const auto& non_fake = GatherVector(
          trivial_loop,
          ExcludeIndex(trivial_loop.size(), pattern.fake_reduce_iter_idx));
      const auto& fake =
          GatherVector(trivial_loop, pattern.fake_reduce_iter_idx);
      return ConcatVector(non_fake, fake);
    }
  }

  MaybeLoopFramework operator()(const UnsupportPattern<T>& pattern) {
    PADDLE_ENFORCE(false, "Not support GetLoopRange.");
  }
};

template <typename T>
MaybeLoopFramework GetLoopFramework(const StmtPattern<T>& pattern) {
  return std::visit(LoopFrameworkVisitor<T>(), pattern.variant());
}

static MaybeLoopFramework SqueezeLoopFramework(
    const MaybeLoopFramework& loop_framework) {
  MaybeLoopFramework result;
  for (int i = 0; i < loop_framework.size(); i++) {
    if (loop_framework[i] == 1) {
      continue;  // skip 1
    } else {
      result.push_back(loop_framework[i]);
    }
  }
  return result;
}

template <typename T>
bool IsLoopFrameworkEqual(const StmtPattern<T>& lhs,
                          const StmtPattern<T>& rhs) {
  auto lhs_loop = GetLoopFramework(lhs);
  auto rhs_loop = GetLoopFramework(rhs);
  VLOG(4) << "lhs loop range is:" << utils::Join(lhs_loop, ",");
  VLOG(4) << "rhs loop range is:" << utils::Join(rhs_loop, ",");
  return SqueezeLoopFramework(lhs_loop) == SqueezeLoopFramework(rhs_loop);
}

template <typename T>
bool IsReducePattern(const StmtPattern<T>& pattern) {
  return std::holds_alternative<ReducePattern<T>>(pattern);
}

template <typename T>
bool IsReduceTreePattern(const StmtPattern<T>& pattern) {
  return std::holds_alternative<ReduceTreePattern<T>>(pattern);
}

template <typename T>
bool IsOpsDependents(const StmtPattern<T>& pattern) {
  return std::holds_alternative<ReduceTreePattern<T>>(pattern);
}

template <typename T>
bool IsUnsupportPattern(const StmtPattern<T>& pattern) {
  return std::holds_alternative<UnsupportPattern<T>>(pattern);
}

template <typename T>
bool IsReduceTrivialPattern(const StmtPattern<T>& pattern) {
  return std::holds_alternative<ReduceTreePlusTrivialPattern<T>>(pattern);
}

template <typename T>
std::unordered_set<pir::Value> GetPatternInputValuesIncludeInner(
    const StmtPattern<T>& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->operands()) {
      result.insert(value.source());
    }
  }
  return result;
}

template <typename T>
std::unordered_set<pir::Value> GetPatternOutputValuesIncludedInner(
    const StmtPattern<T>& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->results()) {
      result.insert(value);
    }
  }
  return result;
}

template <typename T>
std::unordered_set<pir::Value> GetPatternInputValues(const StmtPattern<T>& A) {
  auto all_input_values = GetPatternInputValuesIncludeInner(A);
  for (const auto& value : GetPatternOutputValuesIncludedInner(A)) {
    all_input_values.erase(value);
  }
  VLOG(4) << "GetPatternInputValues: " << all_input_values.size();
  return all_input_values;
}

template <typename T>
std::string StmtPatternDebugStr(const StmtPattern<T>& stmt) {
  std::stringstream ss;
  auto all_ops = GetOpsInPattern(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(all_ops);
  return ss.str();
}

static bool IsDirectUpstream(const pir::Operation* upstream,
                             const pir::Operation* downstream) {
  for (const auto& value : downstream->results()) {
    for (const auto& operand : upstream->operands()) {
      if (value == operand.source()) {
        return true;
      }
    }
  }
  return false;
}

template <typename T>
int InsertDownstreamIntoTree(const ReduceTreePattern<T>& upstream,
                             ReduceTreePattern<T>& downstream) {  // NOLINT
  if (IsDirectUpstream(upstream.GetRootPattern().GetReduceOp(),
                       downstream.GetRootPattern().GetReduceOp())) {
    downstream.InsertChild(upstream);
    return 1;
  }
  int insert_num = 0;
  for (auto& child : downstream.childs()) {
    insert_num += InsertDownstreamIntoTree(upstream, child);
  }
  return insert_num;
}

template <typename T>
StmtPattern<T> MergePatternImpl(const ReduceTreePattern<T>& upstream,
                                const ReduceTreePattern<T>& downstream) {
  ReduceTreePattern<T> result = downstream;  // copy first.
  int insert_num = InsertDownstreamIntoTree(upstream, result);
  CHECK(insert_num == 1) << "Must insert only once, but insert " << insert_num;
  return result;
}

inline auto GetPaddingVector(const MaybeLoopFramework& first,
                             const MaybeLoopFramework& second) {
  // two pointer to get the padding body.
  std::vector<int> padding_f;
  std::vector<int> padding_s;
  VLOG(4) << "GetPaddingVector for: " << utils::Join(first, ",") << " vs "
          << utils::Join(second, ",");
  std::function<void(int, int, int)> RecursivePadding =
      [&first, &second, &padding_f, &padding_s, &RecursivePadding](
          int pf, int ps, int padding_size) {
        if (pf == first.size() && ps == second.size()) {
          return;
        } else if (pf == first.size()) {
          PADDLE_ENFORCE(second[ps] == 1, "second[ps] must be '1' to padding.");
          padding_f.push_back(padding_size);
          RecursivePadding(pf, ps + 1, padding_size + 1);
        } else if (ps == second.size()) {
          PADDLE_ENFORCE(first[pf] == 1, "second[ps] must be '1' to padding.");
          padding_s.push_back(padding_size);
          RecursivePadding(pf + 1, ps, padding_size + 1);
        } else if (second[ps] == first[pf]) {
          RecursivePadding(pf + 1, ps + 1, padding_size + 1);
        } else if (second[ps] == 1) {
          padding_f.push_back(padding_size);
          RecursivePadding(pf, ps + 1, padding_size + 1);
        } else if (first[ps] == 1) {
          padding_s.push_back(padding_size);
          RecursivePadding(pf + 1, ps, padding_size + 1);
        } else {
          PADDLE_THROW("Padding Error.");
        }
      };
  RecursivePadding(0, 0, 0);
  VLOG(4) << "GetPaddingVector result: " << utils::Join(padding_f, ",")
          << " vs " << utils::Join(padding_s, ",");
  return std::tuple(padding_f, padding_s);
}

template <typename T>
StmtPattern<T> MergePatternImpl(const HorizontalFusionPattern<T>& first,
                                const HorizontalFusionPattern<T>& second) {
  const auto& [f, s] =
      GetPaddingVector(GetLoopFramework(StmtPattern<T>(first)),
                       GetLoopFramework(StmtPattern<T>(second)));
                                                                        s};
}

template <typename T>
StmtPattern<T> MergePatternImpl(const ReduceTreePattern<T>& first,
                                const TrivialPattern<T>& second);

template <typename T>
StmtPattern<T> MergePatternImpl(const TrivialPattern<T>& first,
                                const ReducePattern<T>& second);

template <typename T>
StmtPattern<T> MergePatternImpl(const TrivialPattern<T>& first,
                                const TrivialPattern<T>& second);

template <typename T>
StmtPattern<T> MergePatternImpl(const HorizontalFusionPattern<T>& first,
                                const HorizontalFusionPattern<T>& second);

template <typename T>
StmtPattern<T> MergePattern(const StmtPattern<T>& first,
                            const StmtPattern<T>& second) {
  VLOG(4) << "MergePattern: " << GetPatternName(first) << " x "
          << GetPatternName(second);
  const auto PatternMatch = adt::match{
      [&](const ReduceTreePattern<T>& lhs, const ReduceTreePattern<T>& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const ReduceTreePattern<T>& lhs, const TrivialPattern<T>& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern<T>& lhs, const ReducePattern<T>& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern<T>& lhs, const TrivialPattern<T>& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const HorizontalFusionPattern<T>& lhs,
          const HorizontalFusionPattern<T>& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const auto& lhs, const auto& rhs) -> StmtPattern<T> {
        CHECK(false) << "Found not support merge!" << GetPatternName(first)
                     << "X" << GetPatternName(second);
      },
  };
  return std::visit(PatternMatch, first.variant(), second.variant());
}

}  // namespace cinn::fusion
