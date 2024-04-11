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
#include "paddle/cinn/operator_fusion/policy/shardable_axes_base.h"

namespace cinn::fusion {

ShardableAxes ShardableAxesInfoManager::ReplaceShardableAxesWithRootName(
    const ShardableAxes& axes) {
  std::vector<std::string> names;
  for (auto name : axes.axis_names) {
    names.push_back(name_union_[name]);
  }
  return ShardableAxes(names);
}

ShardableAxesSignature ShardableAxesInfoManager::GetSignature(
    pir::Operation* op) {
  return op_signature_map_[op];
  // TODO(baizhou) fix broadcast signature and enable here
  // auto result = ShardableAxesSignature();
  // auto origin_sig = op_signature_map_[op];
  // for (const auto& axes : origin_sig.inputs) {
  //   result.inputs.emplace_back(ReplaceShardableAxesWithRootName(axes));
  // }
  // for (const auto& axes : origin_sig.outputs) {
  //   result.outputs.emplace_back(ReplaceShardableAxesWithRootName(axes));
  // }
  // return result;
}

ShardableAxes ShardableAxesInfoManager::GetAxes(pir::Value value) {
  return ReplaceShardableAxesWithRootName(value_axes_map_[value]);
}

std::string ShardableAxesInfoManager::GetUniqueName() {
  static std::atomic<int64_t> counter = 0;
  counter += 1;
  return "D" + std::to_string(counter);
}

std::vector<std::string> CreateNewNamesWithRank(int64_t rank) {
  auto result = std::vector<std::string>();
  for (int64_t i = 0; i < rank; i++) {
    result.emplace_back(ShardableAxesInfoManager::GetUniqueName());
  }
  return result;
}

ShardableAxesSignature CreateDefaultSignature(pir::Operation* op) {
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
    pir::Operation* op) {
  if (op->isa<cinn::dialect::ReshapeOp>()) {
    return CreateDefaultSignature(op);
  }
  if (op->name() == "cinn_op.generate_shape") {
    return CreateDefaultSignature(op);
  }
  if (op->name() == "cinn_op.yield_store") {
    return CreateDefaultSignature(op);
  }
  if (op->name() == "cinn_op.reshape") {
    return CreateDefaultSignature(op);
  }
  if (op->name() == "pd_op.reshape") {
    return CreateDefaultSignature(op);
  }
  return std::nullopt;
}

ShardableAxesSignature CreateSignatureForReduce(pir::Operation* reduce_op) {
  CHECK_EQ(reduce_op->num_operands(), 1);
  CHECK_EQ(reduce_op->num_results(), 1);
  ShardableAxesSignature result = ShardableAxesSignature();
  const size_t input_rank = GetRank(reduce_op->operand_source(0));
  auto input_axes = CreateNewNamesWithRank(input_rank);

  const auto& reduce_axis_idx = GetReduceAxisIdx(reduce_op);
  bool keep_dim = GetReduceOpKeepDims(reduce_op);
  auto output_axes = std::vector<std::string>();

  if (reduce_axis_idx.empty()) {
    // When reduce_axis is empty, it means all axes are reduced and the output
    // should be a new single axis.
    output_axes.emplace_back(ShardableAxesInfoManager::GetUniqueName());
  } else {
    for (int i = 0; i < input_rank; i++) {
      if (std::find(reduce_axis_idx.begin(), reduce_axis_idx.end(), i) !=
          reduce_axis_idx.end()) {
        if (keep_dim) {
          output_axes.emplace_back(ShardableAxesInfoManager::GetUniqueName());
        }  // else do nothing
      } else {
        output_axes.emplace_back(input_axes[i]);
      }
    }
  }

  result.inputs.emplace_back(input_axes);
  result.outputs.emplace_back(output_axes);

  return result;
}

ShardableAxesSignature CreateSignatureForElementWise(pir::Operation* op) {
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

ShardableAxesSignature CreateSignatureForBroadcast(
    pir::Operation* op, const pir::ShapeConstraintIRAnalysis* shape_analysis) {
  ShardableAxesSignature result = ShardableAxesSignature();

  const auto& broad_cast_value = GetBroadcastOpInputOuputValue(op);
  CHECK(broad_cast_value.has_value());

  const auto& [input_value, output_value] = broad_cast_value.value();
  const int input_rank = GetRank(input_value);
  const int output_rank = GetRank(output_value);
  CHECK_GE(output_rank, input_rank);

  // Create axes for operands. For expand op, the second operand is the shape of
  // output.
  for (int i = 0; i < op->num_operands(); ++i) {
    result.inputs.emplace_back(
        CreateNewNamesWithRank(GetRank(op->operand_source(i))));
  }

  // Create output axes. Compare axis one by one, from back to front.
  // The rule of broadcasting:
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/tensor_cn.html#id7
  const auto& input_axis_names = result.inputs[0].axis_names;
  std::vector<std::string> output_axis_names;
  for (int i = 1; i <= output_rank; ++i) {
    int input_axis = input_rank - i;
    int output_axis = output_rank - i;
    if ((input_axis >= 0) &&
        shape_analysis->IsProductEqual(
            input_value, {input_axis}, output_value, {output_axis})) {
      output_axis_names.emplace_back(input_axis_names[input_axis]);
    } else {
      output_axis_names.emplace_back(ShardableAxesInfoManager::GetUniqueName());
    }
  }
  std::reverse(output_axis_names.begin(), output_axis_names.end());
  result.outputs.emplace_back(ShardableAxes(output_axis_names));

  return result;
}

ShardableAxesSignature ShardableAxesInfoManager::CreateShardableSignature(
    pir::Operation* op) {
  auto special_result = CreateSignatureForSpecialOps(op);
  if (special_result != std::nullopt) {
    VLOG(4) << "[ShardableAxesInfoManager] Create Shardable Axes Signature : \n"
            << op->name() << " : " << special_result.value().DebugStr();
    return special_result.value();
  }

  CHECK(op->num_results() == 1)
      << "Now we do not support op with multi outputs: " << op->name();
  ShardableAxesSignature result;
  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  if (kind == hlir::framework::kReduction) {
    result = CreateSignatureForReduce(op);
  } else if (kind == hlir::framework::kElementWise) {
    result = CreateSignatureForElementWise(op);
  } else if (kind == hlir::framework::kBroadcast) {
    result = CreateSignatureForBroadcast(op, shape_analysis_);
  } else {
    result = CreateDefaultSignature(op);
  }
  VLOG(4) << "[ShardableAxesInfoManager] Create Shardable Axes Signature : \n"
          << op->name() << " : " << result.DebugStr();
  return result;
}

ShardableAxesInfoManager::ShardableAxesInfoManager(
    const std::vector<pir::Operation*>& ops,
    const pir::ShapeConstraintIRAnalysis* shape_analysis)
    : ops_(ops), shape_analysis_(shape_analysis) {
  for (const auto& op : ops) {
    if (op->name() == "cf.yield") continue;
    op_signature_map_[op] = CreateShardableSignature(op);
  }

  const auto FindRoot = [&](std::string non_root) {
    std::string result = non_root;
    while (name_union_[result] != result) {
      result = name_union_[result];
    }
    return result;
  };

  const auto CombineAxes = [&](const ShardableAxes& root,
                               const ShardableAxes& non_root) {
    CHECK_EQ(root.axis_names.size(), non_root.axis_names.size());
    for (int i = 0; i < non_root.axis_names.size(); i++) {
      name_union_[non_root.axis_names[i]] = FindRoot(root.axis_names[i]);
    }
  };

  for (const auto& [op, axes_signature] : op_signature_map_) {
    for (int i = 0; i < op->num_operands(); ++i) {
      auto value = op->operand_source(i);
      auto axes = axes_signature.inputs[i];
      if (value_axes_map_.find(value) == value_axes_map_.end()) {
        value_axes_map_[value] = axes;
        for (auto& axis_name : axes.axis_names) {
          name_union_[axis_name] = axis_name;
        }
      } else {
        CombineAxes(value_axes_map_[value], axes);
      }
    }
    for (int i = 0; i < op->num_results(); ++i) {
      auto value = op->result(i);
      auto axes = axes_signature.outputs[i];
      if (value_axes_map_.find(value) == value_axes_map_.end()) {
        value_axes_map_[value] = axes;
        for (auto& axis_name : axes.axis_names) {
          name_union_[axis_name] = axis_name;
        }
      } else {
        CombineAxes(value_axes_map_[value], axes);
      }
    }
  }

  VLOG(4) << NameUnionDebugStr();
}

std::string ShardableAxes::DebugStr() const {
  std::stringstream ss;
  for (const auto& name : axis_names) {
    ss << name << ", ";
  }
  return ss.str();
}

std::string ShardableAxesSignature::DebugStr() const {
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

std::string ShardableAxesInfoManager::NameUnionDebugStr() const {
  std::stringstream ss;
  ss << "[ShardableAxesInfoManager] NameUnion :\n";

  std::unordered_map<std::string, std::vector<std::string>> root_to_sons;
  for (const auto& [non_root, root] : name_union_) {
    if (root_to_sons.find(root) == root_to_sons.end()) {
      root_to_sons[root] = std::vector<std::string>{non_root};
    } else {
      root_to_sons[root].push_back(non_root);
    }
  }
  for (const auto& [root, sons] : root_to_sons) {
    ss << "Root " << root << ": ";
    for (const auto& son : sons) {
      ss << son << ", ";
    }
    ss << "\n";
  }

  return ss.str();
}

}  // namespace cinn::fusion
