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

#include "paddle/cinn/frontend/group_cluster/pattern_base.h"

namespace cinn::frontend::group_cluster {

bool IsTrivialPattern(const StmtPattern& pattern) {
  return std::holds_alternative<TrivialPattern>(pattern);
}

bool IsReducePattern(const StmtPattern& pattern) {
  return std::holds_alternative<ReducePattern>(pattern);
}

bool IsUnsupportPattern(const StmtPattern& pattern) {
  return std::holds_alternative<UnsupportPattern>(pattern);
}

std::unordered_set<const pir::Operation*> GetOpsInPattern(
    const StmtPattern& pattern) {
  return std::visit([](const auto& impl) { return impl.ops_; }, pattern);
}

std::string StmtPatternDebugStr(const StmtPattern& stmt) {
  std::stringstream ss;
  auto all_ops = GetOpsInPattern(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(
      std::vector<const pir::Operation*>(all_ops.begin(), all_ops.end()));
  return ss.str();
}

std::unordered_set<const pir::Operation*> MergeOpSet(
    const std::unordered_set<const pir::Operation*>& first,
    const std::unordered_set<const pir::Operation*>& second) {
  std::unordered_set<const pir::Operation*> result;
  result.insert(first.begin(), first.end());
  result.insert(second.begin(), second.end());
  return result;
}

StmtPattern MergePattern(const StmtPattern& first, const StmtPattern& second) {
  std::unordered_set<const pir::Operation*> ops =
      MergeOpSet(GetOpsInPattern(first), GetOpsInPattern(second));
  if (IsUnsupportPattern(first) || IsUnsupportPattern(second)) {
    return UnsupportPattern(ops);
  } else if (IsReducePattern(first) || IsReducePattern(second)) {
    return ReducePattern(ops);
  } else {
    return TrivialPattern(ops);
  }
}

StmtPattern ConvertToStmtPattern(const pir::Operation* op) {
  const auto& kind = GetOpPatternKind(op);
  if (kind == hlir::framework::kReduction) {
    return ReducePattern({op});
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    return TrivialPattern({op});
  } else {
    return UnsupportPattern({op});
  }
}

}  // namespace cinn::frontend::group_cluster
