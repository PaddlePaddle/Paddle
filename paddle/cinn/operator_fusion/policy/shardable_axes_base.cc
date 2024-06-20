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

#include "paddle/cinn/operator_fusion/policy/shardable_axes_base.h"

#include "paddle/common/enforce.h"

namespace cinn::fusion {

ShardableAxes ShardableAxesInfoManager::ReplaceShardableAxesWithRootName(
    const ShardableAxes& axes) {
  std::vector<std::string> names;
  const auto FindRoot = [&](std::string non_root) {
    std::string result = non_root;
    while (name_union_[result] != result) {
      result = name_union_[result];
    }
    return result;
  };
  for (auto name : axes.axis_names) {
    names.push_back(FindRoot(name));
  }
  return ShardableAxes(names);
}

ShardableAxesSignature ShardableAxesInfoManager::GetSignature(
    pir::Operation* op) {
  return op_signature_map_[op];
}

ShardableAxesSignature ShardableAxesInfoManager::GetModifiedSignature(
    pir::Operation* op) {
  auto result = ShardableAxesSignature();
  auto origin_sig = op_signature_map_[op];
  for (const auto& axes : origin_sig.inputs) {
    result.inputs.emplace_back(ReplaceShardableAxesWithRootName(axes));
  }
  for (const auto& axes : origin_sig.outputs) {
    result.outputs.emplace_back(ReplaceShardableAxesWithRootName(axes));
  }
  return result;
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
        CreateNewNamesWithRank(GetCompitableRank(op->operand_source(i))));
  }
  for (int i = 0; i < op->num_results(); ++i) {
    result.outputs.emplace_back(
        CreateNewNamesWithRank(GetCompitableRank(op->result(i))));
  }
  return result;
}

std::optional<ShardableAxesSignature> CreateSignatureForSpecialOps(
    pir::Operation* op) {
  if (op->num_results() != 1) {
    VLOG(4) << "Now we do not support op with multi outputs, create default: "
            << op->name();
    return CreateDefaultSignature(op);
  }
  if (op->isa<cinn::dialect::ReshapeOp>()) {
    return CreateDefaultSignature(op);
  }
  if (op->name() == "cinn_op.generate_shape") {
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
  PADDLE_ENFORCE_EQ(
      reduce_op->num_operands(),
      1,
      ::common::errors::PreconditionNotMet(
          "Required reduce_op->num_operands() shall be equal 1."));
  PADDLE_ENFORCE_EQ(reduce_op->num_results(),
                    1,
                    ::common::errors::PreconditionNotMet(
                        "Required reduce_op->num_results() shall be equal 1."));
  ShardableAxesSignature result = ShardableAxesSignature();
  const size_t input_rank = GetCompitableRank(reduce_op->operand_source(0));
  auto input_axes = CreateNewNamesWithRank(input_rank);

  const auto reduce_axis_idx = [&]() -> decltype(auto) {
    const std::vector<int64_t> axis_idx = GetReduceAxisIdx(reduce_op);
    return std::unordered_set<int64_t>(axis_idx.begin(), axis_idx.end());
  }();
  PADDLE_ENFORCE_NE(
      reduce_axis_idx.size(),
      0,
      ::common::errors::PreconditionNotMet(
          "Required reduce_axis_idx.size() shall not be equal 0."));
  bool keep_dim = GetReduceOpKeepDims(reduce_op);
  const auto output_axes = [&]() -> decltype(auto) {
    std::vector<std::string> axes;
    // In case of reduce all and keep_dim is false.
    if (reduce_axis_idx.size() == input_rank && !keep_dim) {
      axes.emplace_back(ShardableAxesInfoManager::GetUniqueName());
      return axes;
    }
    for (size_t i = 0; i < input_rank; i++) {
      if (!reduce_axis_idx.count(i)) {
        axes.emplace_back(input_axes[i]);
      } else if (keep_dim) {
        axes.emplace_back(ShardableAxesInfoManager::GetUniqueName());
      } else {
        // do nothing
      }
    }
    return axes;
  }();

  result.inputs.emplace_back(input_axes);
  result.outputs.emplace_back(output_axes);

  return result;
}

ShardableAxesSignature CreateSignatureForElementWise(pir::Operation* op) {
  ShardableAxesSignature result = ShardableAxesSignature();

  int64_t rank = GetCompitableRank(op->result(0));
  auto same_axes = CreateNewNamesWithRank(rank);

  for (int i = 0; i < op->num_operands(); ++i) {
    PADDLE_ENFORCE_EQ(rank,
                      GetCompitableRank(op->operand_source(i)),
                      ::common::errors::PreconditionNotMet(
                          "Required all inputs rank shall be equal output in "
                          "elementwise op."));
    result.inputs.emplace_back(same_axes);
  }
  for (int i = 0; i < op->num_results(); ++i) {
    PADDLE_ENFORCE_EQ(rank,
                      GetCompitableRank(op->result(i)),
                      ::common::errors::PreconditionNotMet(
                          "Required all outputs rank shall be equal each other "
                          "in elementwise op."));
    result.outputs.emplace_back(same_axes);
  }
  return result;
}

ShardableAxesSignature CreateSignatureForBroadcast(
    pir::Operation* op, pir::ShapeConstraintIRAnalysis* shape_analysis) {
  ShardableAxesSignature result = ShardableAxesSignature();

  const auto& broad_cast_value = GetBroadcastOpInputOuputValue(op);
  PADDLE_ENFORCE_EQ(broad_cast_value.has_value(),
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Required broad_cast_value is not empty."));

  const auto& [input_value, output_value] = broad_cast_value.value();
  const int input_rank = GetCompitableRank(input_value);
  const int output_rank = GetCompitableRank(output_value);
  PADDLE_ENFORCE_GE(
      output_rank,
      input_rank,
      ::common::errors::PreconditionNotMet(
          "Required output rank shall be greater than or equal input rank."));

  // Create axes for operands. For expand op, the second operand is the shape of
  // output.
  for (int i = 0; i < op->num_operands(); ++i) {
    result.inputs.emplace_back(
        CreateNewNamesWithRank(GetCompitableRank(op->operand_source(i))));
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
    VLOG(4) << "[ShardableAxesInfoManager] Create Shardable Axes Signature for "
               "Special Op: \n"
            << op->name() << " : " << special_result.value().DebugStr();
    return special_result.value();
  }

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
    pir::ShapeConstraintIRAnalysis* shape_analysis)
    : ops_(ops), shape_analysis_(shape_analysis) {
  for (const auto& op : ops) {
    if (op->name() == "cf.yield") continue;
    op_signature_map_[op] = CreateShardableSignature(op);
  }

  // short cut
  const auto FindRoot = [&](std::string non_root) {
    std::string result = non_root;
    while (name_union_[result] != result) {
      result = name_union_[result];
    }
    return result;
  };

  const auto CombineAxes = [&](const ShardableAxes& root,
                               const ShardableAxes& non_root) {
    VLOG(4) << "start CombineAxes: " << root.DebugStr() << " with "
            << non_root.DebugStr();
    PADDLE_ENFORCE_EQ(
        root.axis_names.size(),
        non_root.axis_names.size(),
        ::common::errors::PreconditionNotMet(
            "Required root and non_root shall have same size of axis_names."));
    for (int i = 0; i < non_root.axis_names.size(); i++) {
      VLOG(4) << "Link " << non_root.axis_names[i] << " -> "
              << FindRoot(root.axis_names[i]);
      name_union_[FindRoot(non_root.axis_names[i])] =
          FindRoot(root.axis_names[i]);
    }
  };

  // init the name_union_
  for (const auto& [op, axes_signature] : op_signature_map_) {
    for (int i = 0; i < op->num_operands(); ++i) {
      auto value = op->operand_source(i);
      auto axes = axes_signature.inputs[i];
      for (auto& axis_name : axes.axis_names) {
        name_union_[axis_name] = axis_name;
      }
    }
  }

  for (const auto& [op, axes_signature] : op_signature_map_) {
    for (int i = 0; i < op->num_operands(); ++i) {
      auto value = op->operand_source(i);
      auto axes = axes_signature.inputs[i];
      VLOG(4) << op->name() << " " << i << "-th input " << value.impl()
              << " axes: " << axes.DebugStr();
      if (value_axes_map_.find(value) == value_axes_map_.end()) {
        value_axes_map_[value] = axes;
      } else {
        CombineAxes(value_axes_map_[value], axes);
      }
    }
    for (int i = 0; i < op->num_results(); ++i) {
      auto value = op->result(i);
      auto axes = axes_signature.outputs[i];
      VLOG(4) << op->name() << " " << i << "-th output " << value.impl()
              << " axes: " << axes.DebugStr();
      if (value_axes_map_.find(value) == value_axes_map_.end()) {
        value_axes_map_[value] = axes;
      } else {
        CombineAxes(value_axes_map_[value], axes);
      }
    }
  }
  // update the name union.
  for (const auto& [child, father] : name_union_) {
    name_union_[child] = FindRoot(child);
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
