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

bool IsTrivialPattern(const StmtPattern& pattern);
bool IsHorizontalFusionPattern(const StmtPattern& pattern);
bool IsReducePattern(const StmtPattern& pattern);
bool IsReduceTreePattern(const StmtPattern& pattern);
bool IsUnsupportPattern(const StmtPattern& pattern);
bool IsReduceTrivialPattern(const StmtPattern& pattern);

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

std::vector<pir::Operation*> GetOpsInPattern(const StmtPattern& pattern);
std::string StmtPatternDebugStr(const StmtPattern& pattern);
StmtPattern MergePattern(const StmtPattern& first, const StmtPattern& second);
ReducePattern ToReducePattern(const StmtPattern& second);
std::string GetPatternName(const StmtPattern& s);

StmtPattern ConvertToStmtPattern(pir::Operation* op);
std::unordered_set<pir::Value> GetPatternInputValues(const StmtPattern& A);
}  // namespace cinn::frontend::group_cluster
