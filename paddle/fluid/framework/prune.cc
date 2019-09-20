/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/prune.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

const char kFeedOpType[] = "feed";
const char kFetchOpType[] = "fetch";

const char kRecurrent[] = "recurrent";
const char kStates[] = "states";
const char kExStates[] = "ex_states";

bool HasDependentInputVar(
    const proto::OpDesc& op_desc,
    const std::unordered_set<std::string>& dependent_vars) {
  for (auto& var : op_desc.inputs()) {
    for (auto& argu : var.arguments()) {
      if (dependent_vars.count(argu) != 0) {
        return true;
      }
    }
  }
  return false;
}

bool HasDependentOutputVar(
    const proto::OpDesc& op_desc,
    const std::unordered_set<std::string>& dependent_vars) {
  for (auto& var : op_desc.outputs()) {
    for (auto& argu : var.arguments()) {
      if (dependent_vars.count(argu) != 0) {
        return true;
      }
    }
  }
  return false;
}

bool IsTarget(const proto::OpDesc& op_desc) {
  if (op_desc.has_is_target()) {
    return op_desc.is_target();
  }
  return false;
}

bool HasTrueTarget(const proto::OpDesc& op_desc) {
  return op_desc.has_is_target() && op_desc.is_target();
}

bool HasFalseTarget(const proto::OpDesc& op_desc) {
  return op_desc.has_is_target() && !op_desc.is_target();
}

int GetSubBlockIndex(const proto::OpDesc& op_desc) {
  for (auto& attr : op_desc.attrs()) {
    if (attr.type() == proto::AttrType::BLOCK) {
      PADDLE_ENFORCE(attr.has_block_idx());
      return attr.block_idx();
    }
  }
  return -1;
}

bool HasSubBlock(const proto::OpDesc& op_desc) {
  return GetSubBlockIndex(op_desc) > 0;
}

void AppendOpInputVarNames(const proto::OpDesc& op_desc,
                           std::unordered_set<std::string>* vars_set) {
  for (auto& var : op_desc.inputs()) {
    for (auto& arg : var.arguments()) {
      vars_set->emplace(arg);
    }
  }
}

void AppendOpOutputVarNames(const proto::OpDesc& op_desc,
                            std::unordered_set<std::string>* vars_set) {
  for (auto& var : op_desc.outputs()) {
    for (auto& arg : var.arguments()) {
      vars_set->emplace(arg);
    }
  }
}

// block_id is the idx of the current block in the input desc
// parent_block_id is the idx of the parent of the current block
// in the output desc, -1 means the current block is global block
// dependent_vars is passed recursively from the parent block to
// the child block to help pruning
void prune_impl(const proto::ProgramDesc& input, proto::ProgramDesc* output,
                int block_id, int parent_block_id,
                std::unordered_set<std::string>* dependent_vars,
                const std::set<std::string> feed_var_names) {
  auto& block = input.blocks(block_id);
  auto& ops = block.ops();

  bool expect_feed = true;
  for (auto& op_desc : ops) {
    PADDLE_ENFORCE(op_desc.type() != kFeedOpType || expect_feed,
                   "All FeedOps are at the beginning of the ProgramDesc");
    expect_feed = (op_desc.type() == kFeedOpType);
  }

  bool expect_fetch = true;
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    PADDLE_ENFORCE(op_desc.type() != kFetchOpType || expect_fetch,
                   "All FetchOps must at the end of the ProgramDesc");
    expect_fetch = (op_desc.type() == kFetchOpType);
  }

  std::vector<bool> should_run;
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (IsTarget(op_desc) || HasDependentOutputVar(op_desc, *dependent_vars)) {
      // insert its input to the dependency graph
      for (auto& var : op_desc.inputs()) {
        for (auto& argu : var.arguments()) {
          if (feed_var_names.count(argu) == 0) {
            dependent_vars->insert(argu);
          }
        }
      }
      should_run.push_back(true);
    } else {
      should_run.push_back(false);
    }
  }

  // since we are traversing the ProgramDesc in reverse order
  // we reverse the should_run vector
  std::reverse(should_run.begin(), should_run.end());

  // copy the current block from input to output
  auto* block_field = output->mutable_blocks();
  *block_field->Add() = input.blocks(block_id);

  int output_block_id = output->blocks_size() - 1;
  auto* output_block = output->mutable_blocks(output_block_id);
  output_block->set_idx(output_block_id);
  output_block->set_parent_idx(parent_block_id);

  auto* op_field = output_block->mutable_ops();
  op_field->Clear();
  for (size_t i = 0; i < should_run.size(); ++i) {
    if (should_run[i]) {
      auto* op = op_field->Add();
      *op = input.blocks(block_id).ops(i);
      if (HasSubBlock(*op)) {
        VLOG(2) << "Pruning op which has sub block: " << op->type();
        // create sub_block_dependent_vars here to help prune the sub block
        std::unordered_set<std::string> sub_block_dependent_vars;
        for (auto& var : op->inputs()) {
          for (auto& argu : var.arguments()) {
            if (feed_var_names.count(argu) == 0) {
              sub_block_dependent_vars.insert(argu);
            }
          }
        }
        for (auto& var : op->outputs()) {
          for (auto& argu : var.arguments()) {
            if (feed_var_names.count(argu) == 0) {
              sub_block_dependent_vars.insert(argu);
            }
          }
        }

        // Recurrent op's states are also dependent vars
        if (op->type() == kRecurrent) {
          auto& attributes = op->attrs();
          for (auto& attr : attributes) {
            if (attr.name() == kStates || attr.name() == kExStates) {
              for (auto& argu : attr.strings()) {
                if (feed_var_names.count(argu) == 0) {
                  sub_block_dependent_vars.insert(argu);
                }
              }
            }
          }
        }
        // GetSubBlockIndex(*op) is the idx of the sub_block in the input desc
        // output_block_id is the idx of the current block in the output desc
        prune_impl(input, output, GetSubBlockIndex(*op), output_block_id,
                   &sub_block_dependent_vars, feed_var_names);
      }
    }
  }

  // remove the VarDescs in BlockDesc that are not referenced in
  // the pruned OpDescs
  std::unordered_map<std::string, proto::VarDesc> var_map;
  auto* var_field = output->mutable_blocks(output_block_id)->mutable_vars();
  for (const auto& var : *var_field) {
    var_map[var.name()] = var;
  }

  std::set<std::string> var_names;
  for (const auto& op : *op_field) {
    auto& input_field = op.inputs();
    for (auto& input_var : input_field) {
      for (auto& arg : input_var.arguments()) {
        if (var_map.count(arg) != 0) {
          var_names.insert(arg);
        }
      }
    }
    auto& output_field = op.outputs();
    for (auto& output_var : output_field) {
      for (auto& arg : output_var.arguments()) {
        if (var_map.count(arg) != 0) {
          var_names.insert(arg);
        }
      }
    }
  }

  var_field->Clear();
  for (const auto& name : var_names) {
    *var_field->Add() = var_map[name];
  }
}

// TODO(fengjiayi): Prune() could be inplaced to avoid unnecessary copies
void Prune(const proto::ProgramDesc& input,
           const std::set<std::string>& feed_var_names,
           proto::ProgramDesc* output) {
  std::unordered_set<std::string> dependent_vars;
  output->clear_blocks();
  prune_impl(input, output, 0, -1, &dependent_vars, feed_var_names);
}

void CloneWholeBlock(proto::ProgramDesc* input, proto::ProgramDesc* output,
                     int block_id, int parent_block_id) {
  auto* block_field = output->mutable_blocks();
  *block_field->Add() = input->blocks(block_id);
  int output_block_id = output->blocks_size() - 1;
  auto* output_block = output->mutable_blocks(output_block_id);
  output_block->set_idx(output_block_id);
  output_block->set_parent_idx(parent_block_id);
}

void PruneBackwardImpl(proto::ProgramDesc* input, proto::ProgramDesc* output,
                       int block_id, int parent_block_id) {
  // Step 1. Copy the current input block to output
  CloneWholeBlock(input, output, block_id, parent_block_id);
  int output_block_id = output->blocks_size() - 1;
  auto* output_block = output->mutable_blocks(output_block_id);

  // Step 2. Mark forward ops on main branch
  auto* ops = input->mutable_blocks(block_id)->mutable_ops();
  std::unordered_set<std::string> op_input_vars;
  std::unordered_set<std::string> op_output_vars;
  for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (HasTrueTarget(op_desc) ||
        HasDependentOutputVar(op_desc, op_input_vars)) {
      op_desc.set_is_target(true);
      AppendOpInputVarNames(op_desc, &op_input_vars);
      AppendOpOutputVarNames(op_desc, &op_output_vars);
    }
  }

  // Step 3. Mark backward & optimize ops on main branch
  std::unordered_set<std::string> gradop_input_vars;
  std::unordered_set<std::string> gradop_output_vars;
  for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (HasFalseTarget(op_desc) ||
        HasDependentInputVar(op_desc, gradop_output_vars)) {
      op_desc.set_is_target(false);
      AppendOpInputVarNames(op_desc, &gradop_input_vars);
      AppendOpOutputVarNames(op_desc, &gradop_output_vars);
    }
  }

  // Step 4. Mark ops need to be reserved on sub-branch
  for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (!op_desc.has_is_target()) {
      if (HasDependentOutputVar(op_desc, gradop_input_vars)) {
        op_desc.set_is_target(false);
        AppendOpInputVarNames(op_desc, &gradop_input_vars);
      } else {
        op_desc.set_is_target(true);
        AppendOpInputVarNames(op_desc, &op_input_vars);
        AppendOpOutputVarNames(op_desc, &op_output_vars);
      }
    }
  }

  // Step 5. Copy the forward ops to new ProgramDesc
  //   Note: The proto::ProgramDesc doesn't have interface
  //         to remove op and var
  auto* op_field = output_block->mutable_ops();
  op_field->Clear();
  for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
    if (IsTarget(*op_iter)) {
      auto* op = op_field->Add();
      *op = *op_iter;
      if (HasSubBlock(*op)) {
        CloneWholeBlock(input, output, GetSubBlockIndex(*op), output_block_id);
      }
    }
  }

  // Step 6. Copy the forward vars to new ProgramDesc
  // construct all var's map before clear
  auto* var_field = output_block->mutable_vars();
  std::unordered_map<std::string, proto::VarDesc> var_map;
  for (const auto& var : *var_field) {
    var_map[var.name()] = var;
  }
  std::unordered_set<std::string> var_names;
  var_names.insert(op_input_vars.begin(), op_input_vars.end());
  var_names.insert(op_output_vars.begin(), op_output_vars.end());
  var_field->Clear();
  for (const auto& name : var_names) {
    *var_field->Add() = var_map[name];
  }
}

std::unique_ptr<framework::ProgramDesc> PruneBackward(
    const framework::ProgramDesc& origin) {
  // Copy original ProgramDesc, origin can't be change
  framework::ProgramDesc origin_clone(origin);

  // Step 1. Update loss op's role & set loss op to be target
  //   The loss op's op_role is (kForward | kLoss)
  //   The input ProgramDesc should have loss operator.
  auto ops = origin_clone.Block(0).AllOps();
  bool has_loss_op = false;
  for (auto op : ops) {
    int op_role =
        boost::get<int>(op->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    if (op_role == (static_cast<int>(OpRole::kForward) |
                    static_cast<int>(OpRole::kLoss))) {
      op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                  static_cast<int>(OpRole::kForward));
      op->SetIsTarget(true);
      has_loss_op = true;
    } else if (op_role == (static_cast<int>(OpRole::kBackward) |
                           static_cast<int>(OpRole::kLoss))) {
      op->SetIsTarget(false);
      break;
    }
  }
  PADDLE_ENFORCE_EQ(has_loss_op, true,
                    "The Program need to be pruned its backward part"
                    "should have loss operator.");

  // Step 2. Prune backward
  proto::ProgramDesc pruned_desc;
  pruned_desc.clear_blocks();
  PruneBackwardImpl(origin_clone.Proto(), &pruned_desc, 0, -1);

  // Step 3. Contruct new framework::ProgramDesc
  return std::unique_ptr<framework::ProgramDesc>(
      new framework::ProgramDesc(pruned_desc));
}

}  // namespace framework
}  // namespace paddle
