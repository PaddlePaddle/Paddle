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

#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

template <typename T>
ReducePattern<T> ToReducePattern(const StmtPattern<T>& second) {
  return std::get<ReducePattern<T>>(second);
}

template <typename T>
std::string GetPatternName(const StmtPattern<T>& s) {
  return std::visit([](const auto& impl) { return impl.name(); }, s);
}

template <typename T>
StmtPattern<T> ConvertToStmtPattern(const PatternContent<T>& content) {
  CHECK(false) << "Please specialization!";
}

template <typename T>
bool IsHorizontalFusionPattern(const StmtPattern<T>& pattern) {
  return std::holds_alternative<HorizontalFusionPattern<T>>(pattern);
}

template <typename T>
std::vector<pir::Operation*> GetOpsInPattern(const StmtPattern<T>& pattern) {
  return std::visit([](const auto& impl) { return impl.ops(); }, pattern);
}

template <typename T>
bool IsTrivialPattern(const StmtPattern<T>& pattern) {
  return std::holds_alternative<TrivialPattern<T>>(pattern);
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
StmtPattern<T> RT_x_RT(const ReduceTreePattern<T>& upstream,
                       const ReduceTreePattern<T>& downstream) {
  ReduceTreePattern<T> result = downstream;  // copy first.
  int insert_num = InsertDownstreamIntoTree(upstream, result);
  CHECK(insert_num == 1) << "Must insert only once, but insert " << insert_num;
  return result;
}

template <typename T>
StmtPattern<T> RT_x_Trivial(const ReduceTreePattern<T>& first,
                            const TrivialPattern<T>& second) {
  CHECK(false) << "Please specialization!";
}

template <typename T>
StmtPattern<T> Trivial_x_Reduce(const TrivialPattern<T>& first,
                                const ReducePattern<T>& second) {
  CHECK(false) << "Please specialization!";
}

template <typename T>
StmtPattern<T> Trivial_x_Trivial(const TrivialPattern<T>& first,
                                 const TrivialPattern<T>& second) {
  CHECK(false) << "Please specialization!";
}

template <typename T>
StmtPattern<T> H_x_H(const HorizontalFusionPattern<T>& first,
                     const HorizontalFusionPattern<T>& second) {
  CHECK(false) << "Please specialization!";
}

template <typename T>
StmtPattern<T> MergePattern(const StmtPattern<T>& first,
                            const StmtPattern<T>& second) {
  VLOG(4) << "MergePattern: " << GetPatternName(first) << " x "
          << GetPatternName(second);
  VLOG(4) << "MergePattern: " << IsReduceTreePattern<T>(first) << " x "
          << IsTrivialPattern<T>(second);
  if (IsUnsupportPattern(first) || IsUnsupportPattern(second)) {
    CHECK(false) << "Found not support merge!" << GetPatternName(first) << "X"
                 << GetPatternName(second);
  } else if (IsReduceTreePattern<T>(first) && IsReduceTreePattern<T>(second)) {
    return RT_x_RT(std::get<ReduceTreePattern<T>>(first),
                   std::get<ReduceTreePattern<T>>(second));
  } else if (IsReduceTreePattern<T>(first) && IsTrivialPattern<T>(second)) {
    return RT_x_Trivial<T>(std::get<ReduceTreePattern<T>>(first),
                           std::get<TrivialPattern<T>>(second));
  } else if (IsTrivialPattern<T>(first) && IsReducePattern<T>(second)) {
    return Trivial_x_Reduce<T>(std::get<TrivialPattern<T>>(first),
                               std::get<ReducePattern<T>>(second));
  } else if (IsTrivialPattern<T>(first) && IsTrivialPattern<T>(second)) {
    return Trivial_x_Trivial<T>(std::get<TrivialPattern<T>>(first),
                                std::get<TrivialPattern<T>>(second));
  } else if (IsHorizontalFusionPattern<T>(first) &&
             IsHorizontalFusionPattern<T>(second)) {
    return H_x_H<T>(std::get<HorizontalFusionPattern<T>>(first),
                    std::get<HorizontalFusionPattern<T>>(second));
  }
  CHECK(false) << "Found not support merge!" << GetPatternName(first) << "X"
               << GetPatternName(second);
}

}  // namespace cinn::fusion
