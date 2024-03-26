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
#include "paddle/cinn/frontend/group_cluster/cluster_policy/shardable_axes_policy/shardable_axes_base.h"
#include "paddle/cinn/frontend/group_cluster/common_utils.h"

namespace cinn::frontend::group_cluster::policy {

std::string ShardableAxesInfoManager::GetUniqueName() {
  static std::atomic<int64_t> counter = 0;
  return "D" + std::to_string(counter);
}

std::vector<std::string> CreateNewNamesWithRank(int64_t rank) {
  auto result = std::vector<std::string>();
  for (int64_t i = 0; i < rank; i++) {
    result.emplace_back(ShardableAxesInfoManager::GetUniqueName());
  }
  return result;
}

ShardableAxesSignature CreateDefaultSignature(const pir::Operation* op) {
  ShardableAxesSignature result = ShardableAxesSignature();
  for (int i = 0; i < op->num_operands(); ++i) {
    result.inputs.emplace_back(
        CreateNewNamesWithRank(GetRank(op->operand_source(i))));
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.outputs.emplace_back(CreateNewNamesWithRank(GetRank(op->result(i))));
  }
  return result;
}

std::optional<ShardableAxesSignature> CreateSignatureForSpecialOps(
    const pir::Operation* op) {
  if (op->isa<cinn::dialect::ReshapeOp>()) {
    return CreateDefaultSignature(op);
  }
  return std::nullopt;
}

ShardableAxesSignature CreateSignatureForReduce(
    const pir::Operation* reduce_op) {
  CHECK_EQ(reduce_op->num_operands(), 1);
  CHECK_EQ(reduce_op->num_results(), 1);
  ShardableAxesSignature result = ShardableAxesSignature();
  const size_t input_rank = GetRank(reduce_op->operand_source(0));
  auto input_axes = CreateNewNamesWithRank(input_rank);

  const auto& reduce_axis_idx = GetReduceAxisIdx(reduce_op);
  bool keep_dim = GetReduceOpKeepDims(reduce_op);
  auto output_axes = std::vector<std::string>();

  for (int i = 0; i < input_rank; i++) {
    if (std::find(reduce_axis_idx.begin(), reduce_axis_idx.end(), i) !=
        reduce_axis_idx.end()) {
      if (keep_dim) {
        output_axes.emplace_back("constant_1");
      }  // else do nothing
    } else {
      output_axes.emplace_back(input_axes[i]);
    }
  }

  result.inputs.emplace_back(input_axes);
  result.outputs.emplace_back(output_axes);

  return result;
}

ShardableAxesSignature CreateSignatureForElementWise(const pir::Operation* op) {
  ShardableAxesSignature result = ShardableAxesSignature();

  int64_t rank = GetRank(op->result(0));
  auto same_axes = CreateNewNamesWithRank(rank);

  for (int i = 0; i < op->num_operands(); ++i) {
    CHECK(rank == GetRank(op->operand_source(i)));
    result.inputs.emplace_back(same_axes);
  }
  for (int i = 0; i < op->num_results(); ++i) {
    CHECK(rank == GetRank(op->result(i)));
    result.outputs.emplace_back(same_axes);
  }
  return result;
}

ShardableAxesSignature CreateSignatureForBroadcast(const pir::Operation* op) {
  const auto& broad_cast_value = GetBroadcastOpInputOuputValue(op);
  if (!broad_cast_value.has_value()) {
    return CreateDefaultSignature(op);
  }
  const auto& [input, output] = broad_cast_value.value();
  // TODO(wuzhanfei) support broadcast
  return CreateDefaultSignature(op);
}

ShardableAxesSignature CreateShardableSignature(const pir::Operation* op) {
  auto special_result = CreateSignatureForSpecialOps(op);
  if (special_result != std::nullopt) {
    return special_result.value();
  }

  CHECK(op->num_results() == 1)
      << "Now we do not support op with multi outputs";
  ShardableAxesSignature result;
  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  if (kind == hlir::framework::kReduction) {
    result = CreateSignatureForReduce(op);
  } else if (kind == hlir::framework::kElementWise) {
    result = CreateSignatureForElementWise(op);
  } else if (kind == hlir::framework::kBroadcast) {
    result = CreateSignatureForBroadcast(op);
  } else {
    result = CreateDefaultSignature(op);
  }
  VLOG(4) << "[ShardableAxesInfoManager] Create Shardable Axes Signature : \n"
          << op->name() << " : " << result.DebugStr();
  return result;
}

ShardableAxesInfoManager::ShardableAxesInfoManager(
    const std::vector<const pir::Operation*>& ops,
    const pir::ShapeConstraintIRAnalysis* shape_analysis)
    : ops_(ops), shape_analysis_(shape_analysis) {
  for (const auto& op : ops) {
    op_signature_map_[op] = CreateShardableSignature(op);
  }

  // TODO(wuzhanfei) update value_axes_map_ name_union_
}

std::string ShardableAxes::DebugStr() {
  std::stringstream ss;
  for (const auto& name : axis_names) {
    ss << name << ", ";
  }
  return ss.str();
}

std::string ShardableAxesSignature::DebugStr() {
  std::stringstream ss;
  ss << "ShardableAxes Signature:\n";
  for (int i = 0; i < inputs.size(); i++) {
    ss << "input " << i << ": " << inputs[i].DebugStr() << "\n";
  }
  for (int i = 0; i < outputs.size(); i++) {
    ss << "output " << i << ": " << outputs[i].DebugStr() << "\n";
  }
  return ss.str();
}

}  // namespace cinn::frontend::group_cluster::policy
