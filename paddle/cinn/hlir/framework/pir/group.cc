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
  for (auto& iter : this->input_ops) {
    new_group->input_ops[ops_mapper.at(iter.first)] = iter.second;
  }
  for (auto* op : this->output_ops) {
    new_group->output_ops.insert(ops_mapper.at(op));
  }
  for (const auto& output_value : this->output_values) {
    new_group->output_values.push_back(output_value);
  }

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
