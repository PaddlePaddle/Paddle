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
#include <queue>
#include <set>
#include <string>
#include <tuple>
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

int GetOpRole(const proto::OpDesc& op_desc) {
  for (auto& attr : op_desc.attrs()) {
    if (attr.name() == OpProtoAndCheckerMaker::OpRoleAttrName()) {
      PADDLE_ENFORCE(attr.has_i());
      return attr.i();
    }
  }
  return -1;
}

void SetSubBlockIndex(proto::OpDesc* op_desc, int sub_idx) {
  for (auto& attr : *op_desc->mutable_attrs()) {
    if (attr.type() == proto::AttrType::BLOCK) {
      PADDLE_ENFORCE(attr.has_block_idx());
      return attr.set_block_idx(sub_idx);
    }
  }
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

int FindMapByValue(std::map<int, int> m, int val) {
  std::map<int, int>::iterator it;
  for (it = m.begin(); it != m.end(); ++it) {
    if (it->second == val) {
      return it->first;
    }
  }
  return -1;
}

// block_id is the idx of the current block in the input desc
// parent_block_id is the idx of the parent of the current block
// in the output desc, -1 means the current block is global block
// dependent_vars is passed recursively from the parent block to
// the child block to help pruning
void prune_impl(const proto::ProgramDesc& input, proto::ProgramDesc* output,
                int block_id, int parent_block_id,
                std::unordered_set<std::string>* dependent_vars,
                const std::set<std::string> feed_var_names,
                std::map<int, int>* pruned_origin_block_id_map) {
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

  (*pruned_origin_block_id_map)[output_block_id] = block_id;
  std::cout << "map " << output_block_id << " " << block_id << std::endl;
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
                   &sub_block_dependent_vars, feed_var_names,
                   pruned_origin_block_id_map);
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
std::map<int, int> Prune(const proto::ProgramDesc& input,
                         const std::set<std::string>& feed_var_names,
                         proto::ProgramDesc* output) {
  std::unordered_set<std::string> dependent_vars;
  output->clear_blocks();
  std::map<int, int> pruned_origin_block_id_map;
  prune_impl(input, output, 0, -1, &dependent_vars, feed_var_names,
             &pruned_origin_block_id_map);
  // update subblock idx
  for (int i = 0; i < output->blocks_size(); i++) {
    auto* pruned = output->mutable_blocks(i);
    auto* ops = pruned->mutable_ops();
    for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
      auto& op_desc = *op_iter;
      if (HasSubBlock(op_desc)) {
        int origin_sub_idx = GetSubBlockIndex(op_desc);
        std::cout << "idx " << i << " origin_sub_idx " << origin_sub_idx
                  << std::endl;
        auto sub_idx =
            FindMapByValue(pruned_origin_block_id_map, origin_sub_idx);
        PADDLE_ENFORCE_NE(sub_idx, -1,
                          "The origin sub block id should be found in "
                          "pruned_progin_block_id_map");
        SetSubBlockIndex(&op_desc, sub_idx);
      }
    }
  }
  return pruned_origin_block_id_map;
}

void UpdateBlockIdx(proto::ProgramDesc* prog,
                    std::map<int, int> pruned_origin_block_map) {
  return;
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

std::unordered_set<std::string> op_input_vars;
std::unordered_set<std::string> op_output_vars;
std::unordered_set<std::string> gradop_input_vars;
std::unordered_set<std::string> gradop_output_vars;

void PruneBackwardImpl(proto::BlockDesc* origin, proto::BlockDesc* pruned) {
  // Step 2. Mark forward ops on main branch
  auto* ops = origin->mutable_ops();

  for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (HasTrueTarget(op_desc) ||
        HasDependentOutputVar(op_desc, op_input_vars) ||
        HasDependentInputVar(op_desc, op_output_vars)) {
      op_desc.set_is_target(true);
      std::cout << "set true " << op_desc.type() << std::endl;
      AppendOpInputVarNames(op_desc, &op_input_vars);
      AppendOpOutputVarNames(op_desc, &op_output_vars);
    }
  }

  // Step 3. Mark backward & optimize ops on main branch

  for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (HasFalseTarget(op_desc) ||
        HasDependentInputVar(op_desc, gradop_output_vars) ||
        HasDependentOutputVar(op_desc, gradop_output_vars)) {
      op_desc.set_is_target(false);
      std::cout << "set false " << op_desc.type() << std::endl;
      AppendOpInputVarNames(op_desc, &gradop_input_vars);
      AppendOpOutputVarNames(op_desc, &gradop_output_vars);

      if (GetOpRole(op_desc) == static_cast<int>(OpRole::kOptimize)) {
        // remove var both in input and output
        std::cout << "optimize " << op_desc.type() << std::endl;
        std::set<std::string> var_names;
        for (auto& var : op_desc.inputs()) {
          for (auto& arg : var.arguments()) {
            var_names.emplace(arg);
          }
        }
        for (auto& var : op_desc.outputs()) {
          for (auto& arg : var.arguments()) {
            if (var_names.count(arg)) {
              std::cout << arg << std::endl;
              auto iter = gradop_output_vars.find(arg);
              if (iter != gradop_output_vars.end()) {
                gradop_output_vars.erase(iter);
              }
            }
          }
        }
      }
    }
  }

  // Step 4. Mark ops need to be reserved on sub-branch
  for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    if (!op_desc.has_is_target()) {
      if (HasDependentOutputVar(op_desc, gradop_input_vars)) {
        op_desc.set_is_target(false);
        std::cout << "set false " << op_desc.type() << std::endl;
        AppendOpInputVarNames(op_desc, &gradop_input_vars);
      } else {
        op_desc.set_is_target(true);
        std::cout << "set true " << op_desc.type() << std::endl;
        AppendOpInputVarNames(op_desc, &op_input_vars);
        AppendOpOutputVarNames(op_desc, &op_output_vars);
      }
    }
  }

  // Step 5. Copy the forward ops to new ProgramDesc
  //   Note: The proto::ProgramDesc doesn't have interface
  //         to remove op and var
  auto* op_field = pruned->mutable_ops();
  op_field->Clear();
  for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
    if (IsTarget(*op_iter)) {
      auto* op = op_field->Add();
      *op = *op_iter;
    }
  }

  // Step 6. Copy the forward vars to new ProgramDesc
  // construct all var's map before clear
  auto* origin_vars = origin->mutable_vars();
  auto* pruned_vars = pruned->mutable_vars();
  std::unordered_map<std::string, proto::VarDesc> var_map;
  for (const auto& var : *origin_vars) {
    var_map[var.name()] = var;
  }
  pruned_vars->Clear();

  std::unordered_set<std::string> var_names;
  var_names.insert(op_input_vars.begin(), op_input_vars.end());
  var_names.insert(op_output_vars.begin(), op_output_vars.end());
  for (const auto& name : var_names) {
    *pruned_vars->Add() = var_map[name];
    std::cout << name << " " << std::endl;
  }
}

void PruneBackwardImpl2(proto::BlockDesc* origin, proto::BlockDesc* pruned) {
  // Step 3. Mark backward & optimize ops on main branch
  std::unordered_set<std::string> op_input_vars;
  std::unordered_set<std::string> op_output_vars;

  auto* ops = origin->mutable_ops();
  for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
    auto& op_desc = *op_iter;

    auto op_role = GetOpRole(op_desc);
    if (op_role & static_cast<int>(OpRole::kOptimize) ||
        op_role & static_cast<int>(OpRole::kBackward) ||
        op_role & static_cast<int>(OpRole::kLRSched)) {
      op_desc.set_is_target(false);
    }
  }

  // Step 5. Copy the forward ops to new ProgramDesc
  //   Note: The proto::ProgramDesc doesn't have interface
  //         to remove op and var
  auto* op_field = pruned->mutable_ops();
  op_field->Clear();
  for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
    if (!HasFalseTarget(*op_iter)) {
      auto* op = op_field->Add();
      AppendOpInputVarNames(*op_iter, &op_input_vars);
      AppendOpOutputVarNames(*op_iter, &op_output_vars);
      *op = *op_iter;
      std::cout << "true " << op_iter->type() << std::endl;
    }
  }

  // Step 6. Copy the forward vars to new ProgramDesc
  // construct all var's map before clear
  auto* origin_vars = origin->mutable_vars();
  auto* pruned_vars = pruned->mutable_vars();
  std::unordered_map<std::string, proto::VarDesc> var_map;
  for (const auto& var : *origin_vars) {
    std::cout << "origin " << var.name() << std::endl;
    var_map[var.name()] = var;
  }
  pruned_vars->Clear();

  std::unordered_set<std::string> var_names;
  var_names.insert(op_input_vars.begin(), op_input_vars.end());
  var_names.insert(op_output_vars.begin(), op_output_vars.end());
  for (const auto& name : var_names) {
    if (var_map.count(name)) {
      *pruned_vars->Add() = var_map[name];
      std::cout << "cloned " << name << std::endl;
    }
  }
}  // namespace framework

std::tuple<framework::ProgramDesc, std::map<int, int>> PruneBackward(
    const framework::ProgramDesc& origin) {
  // Copy original ProgramDesc, origin can't be change
  framework::ProgramDesc origin_clone(origin);

  // Step 1. Update loss op's role & set loss op to be target
  //   The loss op's op_role is (kForward | kLoss)
  //   The input ProgramDesc should have loss operator.
  bool has_loss_op = false;
  std::queue<int> block_contains_loss;
  std::queue<int> block_contains_loss_grad;
  for (size_t i = 0; i < origin_clone.Size(); i++) {
    auto block_ops = origin_clone.Block(i).AllOps();
    for (auto op : block_ops) {
      int op_role = boost::get<int>(
          op->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
      std::cout << op->Type() << " " << op_role << std::endl;
      if (op_role == (static_cast<int>(OpRole::kForward) |
                      static_cast<int>(OpRole::kLoss))) {
        op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                    static_cast<int>(OpRole::kForward));
        op->SetIsTarget(true);
        std::cout << "true " << op->Type() << std::endl;
        has_loss_op = true;
        block_contains_loss.emplace(i);
      } else if (op_role == (static_cast<int>(OpRole::kBackward) |
                             static_cast<int>(OpRole::kLoss))) {
        op->SetIsTarget(false);
        std::cout << "false " << op->Type() << std::endl;
        block_contains_loss_grad.emplace(i);
        // break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(has_loss_op, true,
                    "The Program need to be pruned its backward part"
                    "should have loss operator.");

  // Step 2. Prune backward
  proto::ProgramDesc pruned_desc;
  pruned_desc.clear_blocks();
  std::vector<proto::BlockDesc*> pruned_blocks;
  std::map<proto::BlockDesc*, proto::BlockDesc*> pruned_progin_block_map;

  for (int i = origin_clone.Size() - 1; i >= 0; i--) {
    auto* pruned = new proto::BlockDesc();
    auto origin = origin_clone.Proto()->mutable_blocks(i);
    PruneBackwardImpl2(origin, pruned);

    for (auto i : op_input_vars) {
      std::cout << " " << i;
    }
    std::cout << std::endl;
    for (auto i : op_output_vars) {
      std::cout << " " << i;
    }
    std::cout << std::endl;
    for (auto i : gradop_input_vars) {
      std::cout << " " << i;
    }
    std::cout << std::endl;
    for (auto i : gradop_output_vars) {
      std::cout << " " << i;
    }
    std::cout << std::endl;

    if (pruned->ops_size() > 0) {
      pruned_blocks.emplace(pruned_blocks.begin(), pruned);
      std::cout << pruned->ops_size() << std::endl;
      pruned_progin_block_map[pruned] = origin;
      std::cout << pruned << " " << origin << std::endl;
    } else {
      delete pruned;
    }
  }
  std::map<int, int> pruned_progin_block_id_map;

  // update idx
  for (size_t i = 0; i < pruned_blocks.size(); i++) {
    auto* pruned = pruned_blocks[i];
    std::cout << pruned << std::endl;
    auto* origin = pruned_progin_block_map[pruned];
    std::cout << origin << std::endl;
    pruned->set_idx(i);
    pruned_progin_block_id_map[i] = origin->idx();
    std::cout << "map " << i << " " << origin->idx() << std::endl;
  }
  std::cout << "update idx done" << std::endl;

  // update parent idx
  for (size_t i = 0; i < pruned_blocks.size(); i++) {
    auto* pruned = pruned_blocks[i];
    auto* origin = pruned_progin_block_map[pruned];
    int parent_idx = -1;
    if (origin->parent_idx() == -1) {
      parent_idx = -1;
    } else {
      parent_idx =
          FindMapByValue(pruned_progin_block_id_map, origin->parent_idx());
      PADDLE_ENFORCE_NE(parent_idx, -1,
                        "The parent block id should be found in "
                        "pruned_progin_block_id_map");
    }
    pruned->set_parent_idx(parent_idx);
  }
  std::cout << "update parenet done" << std::endl;

  // update subblock attrs
  for (size_t i = 0; i < pruned_blocks.size(); i++) {
    auto* pruned = pruned_blocks[i];
    auto* ops = pruned->mutable_ops();
    for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
      auto& op_desc = *op_iter;
      if (HasSubBlock(op_desc)) {
        int origin_sub_idx = GetSubBlockIndex(op_desc);
        auto sub_idx =
            FindMapByValue(pruned_progin_block_id_map, origin_sub_idx);
        PADDLE_ENFORCE_NE(sub_idx, -1,
                          "The origin sub block id should be found in "
                          "pruned_progin_block_id_map");
        SetSubBlockIndex(&op_desc, sub_idx);
      }
    }
  }
  std::cout << "update subblock done" << std::endl;

  // clone blocks to program
  for (size_t i = 0; i < pruned_blocks.size(); i++) {
    auto* block_field = pruned_desc.mutable_blocks();
    *block_field->Add() = *pruned_blocks[i];
    delete pruned_blocks[i];
  }
  std::cout << "clone done" << std::endl;
  for (auto pair : pruned_progin_block_id_map) {
    std::cout << pair.first << " " << pair.second << std::endl;
  }
  // Step 3. Contruct new framework::ProgramDesc

  return std::make_tuple(framework::ProgramDesc(pruned_desc),
                         pruned_progin_block_id_map);
}  // namespace framework

}  // namespace framework
}  // namespace paddle
