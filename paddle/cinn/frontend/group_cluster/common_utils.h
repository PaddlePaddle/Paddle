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

#include "paddle/cinn/frontend/group_cluster/pattern.h"

#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/common/topo_walker.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn::frontend::group_cluster {

using OpPatternKind = cinn::hlir::framework::OpPatternKind;

OpPatternKind GetOpPatternKind(const ::pir::Operation* op);
size_t GetRank(pir::Value value);
std::vector<int64_t> GetReduceAxisIdx(pir::Operation* reduce_op);
bool GetReduceOpKeepDims(pir::Operation* reduce_op);
std::string OpsDebugStr(std::vector<pir::Operation*> ops);
std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOuputValue(
    pir::Operation* op);
}  // namespace cinn::frontend::group_cluster

namespace cinn::frontend::group_cluster {

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
std::vector<PatternContent<T>> GetContentsInPattern(
    const StmtPattern<T>& pattern) {
  return std::visit([](const auto& impl) { return impl.contents(); }, pattern);
}

template <typename T>
pir::Operation* GetOpFromContent(const PatternContent<T>& content) {
  return content.op;
}

template <typename T>
std::vector<pir::Operation*> GetOpsInPattern(const StmtPattern<T>& pattern) {
  std::function<pir::Operation*(PatternContent<T>)> func = GetOpFromContent<T>;
  return MapVector(GetContentsInPattern(pattern), func);
}

template <typename T>
std::string StmtPatternDebugStr(const StmtPattern<T>& stmt) {
  std::stringstream ss;
  auto all_ops = GetOpsInPattern(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(all_ops);
  return ss.str();
}

template <typename T>
StmtPattern<T> MergePattern(const StmtPattern<T>& first,
                            const StmtPattern<T>& second) {
  std::vector<PatternContent<T>> contents =
      MergeVector(GetContentsInPattern(first), GetContentsInPattern(second));
  if (IsUnsupportPattern(first) || IsUnsupportPattern(second)) {
    return UnsupportPattern<T>(contents);
  } else if (IsReduceTreePattern(first) && IsReduceTreePattern(second)) {
    const auto& merged =
        ConcatVector(std::get<ReduceTreePattern<T>>(first).reduce_patterns(),
                     std::get<ReduceTreePattern<T>>(second).reduce_patterns());
    return ReduceTreePattern<T>(
        merged, std::get<ReduceTreePattern<T>>(second).GetRootPattern());
  } else if (IsReduceTreePattern(first) && IsTrivialPattern(second)) {
    return ReduceTreePlusTrivialPattern<T>(
        std::get<ReduceTreePattern<T>>(first),
        std::get<TrivialPattern<T>>(second));
  } else if (IsTrivialPattern(first) && IsReducePattern(second)) {
    return ReducePattern<T>(contents);
  } else if (IsTrivialPattern(first) && IsTrivialPattern(second)) {
    return TrivialPattern<T>(contents);
  } else if (IsHorizontalFusionPattern(first) &&
             IsHorizontalFusionPattern(second)) {
    return HorizontalFusionPattern<T>({first, second});
  } else {
    // Not Implementation.
    CHECK(false) << "Found not support merge!";
  }
}

template <typename T>
bool IsHorizontalFusionPattern(const StmtPattern<T>& pattern) {
  return std::holds_alternative<HorizontalFusionPattern<T>>(pattern);
}

template <typename T>
StmtPattern<T> ConvertToStmtPattern(const PatternContent<T>& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    return ReducePattern<T>({content});
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    return TrivialPattern<T>({content});
  } else {
    return UnsupportPattern<T>({content});
  }
}

template <typename T>
ReducePattern<T> ToReducePattern(const StmtPattern<T>& second) {
  return std::get<ReducePattern<T>>(second);
}

template <typename T>
void RemoveFromVector(std::vector<T>* vec, T item) {
  auto iter = std::find(vec->begin(), vec->end(), item);
  if (iter != vec->end()) {
    vec->erase(iter);
  }
}

template <typename T>
std::vector<T> ConcatVector(const std::vector<T>& first,
                            const std::vector<T>& second) {
  std::vector<T> result = first;
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

template <typename T, typename F>
std::vector<T> FilterVector(const std::vector<T>& first, const F& func) {
  std::vector<T> result;
  for (const auto& i : first) {
    if (func(i)) {
      result.push_back(i);
    }
  }
  return result;
}

template <class A, class B>
std::vector<B> MapVector(const std::vector<A>& as,
                         const std::function<B(A)>& func) {
  std::vector<B> res;
  for (const auto& a : as) {
    res.push_back(func(a));
  }
  return res;
}

template <typename T>
std::set<T> ToSet(const std::vector<T>& input) {
  std::set<T> result(input.begin(), input.end());
  return result;
}

template <typename T>
bool IsAnyFirstInSecond(const std::vector<T>& first,
                        const std::vector<T>& second) {
  const auto& second_set = ToSet(second);
  for (const auto& ele : first) {
    if (second_set.count(ele)) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::vector<T> UniqueVectorBySet(const std::vector<T>& v) {
  std::set<T> unique(v.begin(), v.end());
  return std::vector<T>(unique.begin(), unique.end());
}

template <typename T>
std::string GetPatternName(const StmtPattern<T>& s) {
  return std::visit([](const auto& impl) { return impl.name(); }, s);
}
}  // namespace cinn::frontend::group_cluster
