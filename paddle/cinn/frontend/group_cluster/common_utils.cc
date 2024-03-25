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

std::vector<int64_t> GetReduceAxisIdx(const pir::Operation* reduce_op) {
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
  return reduce_axis_idx;
}

bool GetReduceOpKeepDims(const pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keep_dim");
  CHECK(attr_val.isa<::pir::BoolAttribute>());
  return attr_val.dyn_cast<::pir::BoolAttribute>();
}

std::string OpsDebugStr(std::vector<const pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(const_cast<pir::Operation*>(op));
    ss << "\n";
  }
  return ss.str();
}

std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOuputValue(
    const pir::Operation* op) {
  auto* mut_op = const_cast<pir::Operation*>(op);
  if (op->isa<paddle::dialect::ExpandOp>()) {
    auto expand_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();
    return std::make_pair(expand_op.x(), expand_op.out());
  }
  if (op->isa<cinn::dialect::BroadcastOp>()) {
    auto broadcast_op = mut_op->dyn_cast<cinn::dialect::BroadcastOp>();
    return std::make_pair(broadcast_op.x(), broadcast_op.out());
  }
  VLOG(4) << "[ShardableAxesSignature] Unsupported Broadcast op: "
          << op->name();
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

bool IsUnsupportPattern(const StmtPattern& pattern) {
  return std::holds_alternative<UnsupportPattern>(pattern);
}

std::vector<const pir::Operation*> GetOpsInPattern(const StmtPattern& pattern) {
  return std::visit([](const auto& impl) { return impl.ops_; }, pattern);
}

std::string StmtPatternDebugStr(const StmtPattern& stmt) {
  std::stringstream ss;
  auto all_ops = GetOpsInPattern(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(all_ops);
  return ss.str();
}

StmtPattern MergePattern(const StmtPattern& first, const StmtPattern& second) {
  std::vector<const pir::Operation*> ops =
      MergeVector(GetOpsInPattern(first), GetOpsInPattern(second));
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
