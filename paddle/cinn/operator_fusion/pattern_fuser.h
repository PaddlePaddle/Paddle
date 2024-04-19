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
