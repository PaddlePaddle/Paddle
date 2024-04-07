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

#include "paddle/cinn/frontend/group_cluster/common_utils.h"

namespace cinn::frontend::group_cluster {

OpPatternKind GetOpPatternKind(const ::pir::Operation* op) {
  return hlir::framework::pir::CompatibleInfo::OpKind(*op);
}

size_t GetRank(pir::Value value) {
  return value.type().dyn_cast<pir::DenseTensorType>().dims().size();
}

std::vector<int64_t> GetReduceAxisIdx(pir::Operation* reduce_op) {
  const size_t input_rank = GetRank(reduce_op->operand_source(0));
  const auto& attr_val = reduce_op->attributes().at("dim");
  CHECK(attr_val.isa<::pir::ArrayAttribute>());
  const auto& axis_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  std::vector<int64_t> reduce_axis_idx;
  for (int i = 0; i < axis_attr.size(); ++i) {
    int64_t axis = axis_attr.at(i).dyn_cast<::pir::Int64Attribute>().data();
    if (axis < 0) {
      axis += input_rank;
    }
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_rank);
    reduce_axis_idx.push_back(axis);
  }
  VLOG(4) << "GetReduceAxisIdx: " << utils::Join(reduce_axis_idx, ",");
  return reduce_axis_idx;
}

bool GetReduceOpKeepDims(pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keep_dim");
  CHECK(attr_val.isa<::pir::BoolAttribute>());
  return attr_val.dyn_cast<::pir::BoolAttribute>().data();
}

std::string GetPatternName(const StmtPattern& s) {
  return std::visit([](const auto& impl) { return impl.name(); }, s);
}

std::string OpsDebugStr(std::vector<pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(const_cast<pir::Operation*>(op));
    ss << "\n";
  }
  return ss.str();
}

std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOuputValue(
    pir::Operation* op) {
  auto* mut_op = const_cast<pir::Operation*>(op);
  if (op->isa<paddle::dialect::ExpandOp>()) {
    auto expand_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();
    return std::make_pair(expand_op.x(), expand_op.out());
  } else if (op->isa<cinn::dialect::BroadcastOp>()) {
    auto broadcast_op = mut_op->dyn_cast<cinn::dialect::BroadcastOp>();
    return std::make_pair(broadcast_op.x(), broadcast_op.out());
  } else {
    CHECK(false) << "Unsupported broadcast op: " << op->name();
  }
  return std::nullopt;
}
}  // namespace cinn::frontend::group_cluster

namespace cinn::frontend::group_cluster {

bool IsTrivialPattern(const StmtPattern& pattern) {
  return std::holds_alternative<TrivialPattern>(pattern);
}

bool IsReducePattern(const StmtPattern& pattern) {
  return std::holds_alternative<ReducePattern>(pattern);
}

bool IsReduceTreePattern(const StmtPattern& pattern) {
  return std::holds_alternative<ReduceTreePattern>(pattern);
}

bool IsOpsDependents(const StmtPattern& pattern) {
  return std::holds_alternative<ReduceTreePattern>(pattern);
}

bool IsUnsupportPattern(const StmtPattern& pattern) {
  return std::holds_alternative<UnsupportPattern>(pattern);
}

bool IsReduceTrivialPattern(const StmtPattern& pattern) {
  return std::holds_alternative<ReduceTreePlusTrivialPattern>(pattern);
}

std::unordered_set<pir::Value> GetPatternInputValuesIncludeInner(
    const StmtPattern& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->operands()) {
      result.insert(value.source());
    }
  }
  return result;
}

std::unordered_set<pir::Value> GetPatternOutputValuesIncludedInner(
    const StmtPattern& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->results()) {
      result.insert(value);
    }
  }
  return result;
}

std::unordered_set<pir::Value> GetPatternInputValues(const StmtPattern& A) {
  auto all_input_values = GetPatternInputValuesIncludeInner(A);
  for (const auto& value : GetPatternOutputValuesIncludedInner(A)) {
    all_input_values.erase(value);
  }
  VLOG(4) << "GetPatternInputValues: " << all_input_values.size();
  return all_input_values;
}

std::vector<pir::Operation*> GetOpsInPattern(const StmtPattern& pattern) {
  return std::visit([](const auto& impl) { return impl.ops(); }, pattern);
}

std::string StmtPatternDebugStr(const StmtPattern& stmt) {
  std::stringstream ss;
  auto all_ops = GetOpsInPattern(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(all_ops);
  return ss.str();
}

StmtPattern MergePattern(const StmtPattern& first, const StmtPattern& second) {
  std::vector<pir::Operation*> ops =
      MergeVector(GetOpsInPattern(first), GetOpsInPattern(second));
  if (IsUnsupportPattern(first) || IsUnsupportPattern(second)) {
    return UnsupportPattern(ops);
  } else if (IsReduceTreePattern(first) && IsReduceTreePattern(second)) {
    const auto& merged =
        ConcatVector(std::get<ReduceTreePattern>(first).reduce_patterns_,
                     std::get<ReduceTreePattern>(second).reduce_patterns_);
    return ReduceTreePattern(
        merged, std::get<ReduceTreePattern>(second).GetRootPattern());
  } else if (IsReduceTreePattern(first) && IsTrivialPattern(second)) {
    return ReduceTreePlusTrivialPattern(std::get<ReduceTreePattern>(first),
                                        std::get<TrivialPattern>(second));
  } else if (IsTrivialPattern(first) && IsReducePattern(second)) {
    return ReducePattern(ops);
  } else if (IsTrivialPattern(first) && IsTrivialPattern(second)) {
    return TrivialPattern(ops);
  } else if (IsHorizontalFusionPattern(first) &&
             IsHorizontalFusionPattern(second)) {
    return HorizontalFusionPattern(ops);
  } else {
    // Not Implementation.
    CHECK(false) << "Found not support merge!";
  }
}

bool IsHorizontalFusionPattern(const StmtPattern& pattern) {
  return std::holds_alternative<HorizontalFusionPattern>(pattern);
}

StmtPattern ConvertToStmtPattern(pir::Operation* op) {
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

ReducePattern ToReducePattern(const StmtPattern& second) {
  return std::get<ReducePattern>(second);
}

}  // namespace cinn::frontend::group_cluster
