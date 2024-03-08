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

#include "paddle/cinn/hlir/framework/pir/group.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

std::shared_ptr<Group> Group::Clone(::pir::Block* target_block,
                                    ::pir::IrMapping& ir_mapping,
                                    const Options& option) const {
  CHECK_EQ(option.OnlyCloneOps(), true)
      << "Only Support Clone Group ops information.";
  std::vector<::pir::Operation*> new_ops;
  // Mapper from original to new ops.
  std::unordered_map<::pir::Operation*, ::pir::Operation*> ops_mapper;
  auto clone_options = ::pir::CloneOptions(false, true, false);

  for (auto* op : ops) {
    VLOG(4) << "clone op :" << op->name();
    auto* new_op = op->Clone(ir_mapping, clone_options);
    // NOTE(dev): Must call block.insert to deal with ownership, otherwise it
    // will lead memory-leak.
    target_block->insert(target_block->end(), new_op);
    new_ops.push_back(new_op);
    ops_mapper[op] = new_op;
  }

  // Construct Base information for new Group
  auto new_group = std::make_shared<Group>(new_ops);

  // clone data map expr
  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs> temp_map;
  for (size_t i = 0; i < ops.size(); ++i) {
    for (size_t j = 0; j < ops[i]->num_operands(); ++j) {
      if (value_to_shape_or_data_exprs_.count(ops[i]->operand_source(j))) {
        temp_map.emplace(
            new_ops[i]->operand_source(j),
            value_to_shape_or_data_exprs_.at(ops[i]->operand_source(j)));
      }
    }

    for (size_t j = 0; j < ops[i]->num_regions(); ++j) {
      if (value_to_shape_or_data_exprs_.count(ops[i]->result(j))) {
        temp_map.emplace(new_ops[i]->result(j),
                         value_to_shape_or_data_exprs_.at(ops[i]->result(j)));
      }
    }
  }

  new_group->set_value_to_shape_or_data_exprs(temp_map);

  for (auto& iter : this->input_ops) {
    new_group->input_ops[ops_mapper.at(iter.first)] = iter.second;
  }
  for (auto* op : this->output_ops) {
    new_group->output_ops.insert(ops_mapper.at(op));
  }
  for (const auto& output_value : this->output_values) {
    new_group->output_values.push_back(ir_mapping.Lookup(output_value));
  }

  new_group->input_names = this->input_names;
  new_group->output_names = this->output_names;
  new_group->fn_name = this->fn_name;
  new_group->int_args_map = this->int_args_map;
  new_group->alignment_schedule_info = this->alignment_schedule_info;
  new_group->reduce_axis = this->reduce_axis;
  new_group->loop_ranges = this->loop_ranges;

  return new_group;
}

std::ostream& operator<<(std::ostream& os, const Group& group) {
  ::pir::IrPrinter printer(os);
  os << "Group " << group.group_id << " :\n";
  for (auto* op : group.ops) {
    printer.PrintOperation(op);
    os << "\n";
  }
  return os;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
