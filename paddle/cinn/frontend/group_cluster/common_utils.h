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
std::vector<int64_t> GetReduceAxisIdx(const pir::Operation* reduce_op);
bool GetReduceOpKeepDims(const pir::Operation* reduce_op);
std::string OpsDebugStr(std::vector<const pir::Operation*> ops);
std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOuputValue(
    const pir::Operation* op);
}  // namespace cinn::frontend::group_cluster

namespace cinn::frontend::group_cluster {

bool IsTrivialPattern(const StmtPattern& pattern);
bool IsReducePattern(const StmtPattern& pattern);
bool IsUnsupportPattern(const StmtPattern& pattern);

template <typename T>
void ExtendVector(std::vector<T>* first, const std::vector<T>& second) {
  std::unordered_set<T> visited =
      std::unordered_set<T>(first->begin(), first->end());
  for (auto iter = second.begin(); iter != second.end(); iter++) {
    if (visited.find(*iter) == visited.end()) {
      visited.emplace(*iter);
      first->emplace_back(*iter);
    }
  }
}

template <typename T>
std::vector<T> MergeVector(const std::vector<T>& first,
                           const std::vector<T>& second) {
  std::vector<T> result = std::vector<T>(first);
  ExtendVector(&result, second);
  return result;
}

std::vector<const pir::Operation*> GetOpsInPattern(const StmtPattern& pattern);
std::string StmtPatternDebugStr(const StmtPattern& pattern);
StmtPattern MergePattern(const StmtPattern& first, const StmtPattern& second);

StmtPattern ConvertToStmtPattern(const pir::Operation* op);
}  // namespace cinn::frontend::group_cluster
