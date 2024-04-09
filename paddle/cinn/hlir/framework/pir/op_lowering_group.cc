// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

::pir::Program* OpLoweringGroup::GetParentProgram() const {
  PADDLE_ENFORCE_GT(ops_.size(),
                    0,
                    ::common::errors::PreconditionNotMet(
                        "Require at least one op in the group."));
  PADDLE_ENFORCE_NOT_NULL(
      ops_[0],
      ::common::errors::Unavailable("Found group.ops_[0] is nullptr."));
  return ops_[0]->GetParentProgram();
}

cinn::dialect::FusionOp OpLoweringGroup::FusionOp() const {
  PADDLE_ENFORCE_GT(this->ops_.size(),
                    0,
                    ::common::errors::PreconditionNotMet(
                        "Required at least one operation in OpLoweringGroup."));
  auto* parent_op = this->ops_[0]->GetParentOp();
  PADDLE_ENFORCE(parent_op && parent_op->isa<cinn::dialect::FusionOp>(),
                 ::common::errors::Unavailable(
                     "Required inner op's parent must be FusionOp."));
  return parent_op->dyn_cast<cinn::dialect::FusionOp>();
}

std::vector<::pir::Value> OpLoweringGroup::GetGroupOutputValues() const {
  std::unordered_set<::pir::Operation*> group_ops_set(this->ops_.begin(),
                                                      this->ops_.end());

  std::vector<::pir::Value> output_values;
  for (auto* op : this->ops_) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);
      if (!result) {
        continue;
      }
      for (auto use_iter = result.use_begin(); use_iter != result.use_end();
           ++use_iter) {
        auto* use_op = use_iter->owner();
        if (group_ops_set.find(use_op) == group_ops_set.end()) {
          output_values.push_back(result);
          break;
        }
      }
    }
  }
  return output_values;
}

std::unordered_set<::pir::Value> OpLoweringGroup::GetInputOpValues() const {
  std::unordered_set<::pir::Value> group_inputs;
  std::unordered_set<::pir::Operation*> ops_set(this->ops_.begin(),
                                                this->ops_.end());

  // count all op's input Value
  for (auto op : ops_set) {
    for (auto& value : op->operands_source()) {
      if (!value || !value.type() || ops_set.count(value.defining_op()))
        continue;
      // if the input value owner op is not in OpSet, it's the group's input
      group_inputs.insert(value);
    }
  }
  return group_inputs;
}

std::unordered_set<::pir::Value> OpLoweringGroup::GetOutputOpValues() const {
  std::unordered_set<::pir::Value> group_outputs;

  for (auto op : this->output_ops_) {
    for (auto& result : op->results()) {
      if (!result || result.type()) {
        continue;
      }

      group_outputs.insert(result);
    }
  }
  return group_outputs;
}

const symbol::ShapeOrDataDimExprs& OpLoweringGroup::GetShapeOrDataExprs(
    const ::pir::Value& value) const {
  PADDLE_ENFORCE(HasShapeOrDataExprs(value),
                 ::common::errors::NotFound(
                     "value not found in value_to_shape_or_data_exprs_"));
  return value_to_shape_or_data_exprs_.at(value);
}

void OpLoweringGroup::SetShapeOrDataExprs(
    const ::pir::Value& value,
    const symbol::ShapeOrDataDimExprs& shape_or_data) {
  auto iter = value_to_shape_or_data_exprs_.find(value);
  if (iter == value_to_shape_or_data_exprs_.end()) {
    value_to_shape_or_data_exprs_.emplace(value, shape_or_data);
  } else {
    iter->second = shape_or_data;
  }
}

std::shared_ptr<OpLoweringGroup> OpLoweringGroup::Clone(
    ::pir::Block* target_block, ::pir::IrMapping* ir_mapping) const {
  std::vector<::pir::Operation*> new_ops;
  // Mapper from original to new ops.
  std::unordered_map<::pir::Operation*, ::pir::Operation*> ops_mapper;
  auto clone_options = ::pir::CloneOptions(false, true, false);
  for (auto* op : ops_) {
    VLOG(4) << "clone op :" << op->name();
    auto* new_op = op->Clone(*ir_mapping, clone_options);
    // NOTE(dev): Must call block.insert to deal with ownership, otherwise it
    // will lead memory-leak.
    target_block->insert(target_block->end(), new_op);
    new_ops.push_back(new_op);
    ops_mapper[op] = new_op;
  }

  // Construct Base information for new Group
  auto new_group = std::make_shared<OpLoweringGroup>(new_ops);
  for (auto* op : this->output_ops_) {
    new_group->output_ops_.insert(ops_mapper.at(op));
  }
  for (const auto& output_value : this->output_values_) {
    new_group->output_values_.push_back(ir_mapping->Lookup(output_value));
  }

  new_group->input_names_ = this->input_names_;
  new_group->output_names_ = this->output_names_;
  new_group->int_args_map_ = this->int_args_map_;
  new_group->alignment_schedule_info_ = this->alignment_schedule_info_;
  new_group->reduce_axis_ = this->reduce_axis_;
  new_group->loop_ranges_ = this->loop_ranges_;
  return new_group;
}

std::ostream& operator<<(std::ostream& os, const OpLoweringGroup& group) {
  auto PrintSymbolDims = [&](const ::pir::Operation& op) {
    if (group.value_to_shape_or_data_exprs_.empty()) return;
    os << " {";
    for (uint32_t i = 0; i < op.num_operands(); ++i) {
      if (i > 0) os << ",";
      if (group.HasShapeOrDataExprs(op.operand_source(i))) {
        os << "<" << group.GetShapeOrDataExprs(op.operand_source(i)) << ">";
      }
    }
    os << "} -> {";
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (i > 0) os << ",";
      if (group.HasShapeOrDataExprs(op.result(i))) {
        os << "<" << group.GetShapeOrDataExprs(op.result(i)) << ">";
      }
    }
    os << "}";
  };
  ::pir::IrPrinter printer(os);
  os << "Group " << group.group_id() << " :\n";
  for (auto* op : group.ops()) {
    printer.PrintOperation(op);
    PrintSymbolDims(*op);
    os << "\n";
  }
  return os;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
