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

#include <queue>

#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle::framework {

const char kFeedOpType[] = "feed";    // NOLINT
const char kFetchOpType[] = "fetch";  // NOLINT

const char kRecurrent[] = "recurrent";  // NOLINT
const char kStates[] = "states";        // NOLINT
const char kExStates[] = "ex_states";   // NOLINT

const char kPyLayer[] = "pylayer";  // NOLINT

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
  // The block index >= 0, so -1 is used to indicate "NotFound".
  for (auto& attr : op_desc.attrs()) {
    if (attr.type() == proto::AttrType::BLOCK) {
      PADDLE_ENFORCE_EQ(attr.has_block_idx(),
                        true,
                        common::errors::NotFound(
                            "Attribute sub_block is not found in operator %s",
                            op_desc.type()));
      return attr.block_idx();
    }
  }
  return -1;
}

void GetSubBlocksIndices(const proto::OpDesc& op_desc,
                         std::vector<int>* indices) {
  for (auto& attr : op_desc.attrs()) {
    if (attr.type() == proto::AttrType::BLOCKS) {
      PADDLE_ENFORCE_GT(
          attr.blocks_idx_size(),
          0,
          common::errors::NotFound(
              "Attribute blocks is not found in operator %s", op_desc.type()));
      indices->resize(attr.blocks_idx_size());
      for (int i = 0; i < attr.blocks_idx_size(); i++) {
        (*indices)[i] = attr.blocks_idx(i);
      }
    }
  }
}

void SetSubBlockIndex(proto::OpDesc* op_desc, int sub_idx) {
  for (auto& attr : *op_desc->mutable_attrs()) {
    if (attr.type() == proto::AttrType::BLOCK) {
      PADDLE_ENFORCE_EQ(attr.has_block_idx(),
                        true,
                        common::errors::NotFound(
                            "Attribute sub_block is not found in operator %s",
                            op_desc->type()));
      attr.set_block_idx(sub_idx);
    }
  }
}

void SetSubBlocksIndices(proto::OpDesc* op_desc,
                         const std::vector<int>& sub_indices) {
  for (auto& attr : *op_desc->mutable_attrs()) {
    if (attr.type() == proto::AttrType::BLOCKS) {
      PADDLE_ENFORCE_GT(
          attr.blocks_idx_size(),
          0,
          common::errors::NotFound(
              "Attribute blocks is not found in operator %s", op_desc->type()));
      attr.clear_blocks_idx();
      for (auto idx : sub_indices) {
        attr.add_blocks_idx(idx);
      }
    }
  }
}

bool HasSubBlock(const proto::OpDesc& op_desc) {
  return GetSubBlockIndex(op_desc) > 0;
}

bool HasSubBlocks(const proto::OpDesc& op_desc) {
  // ``blocks_idx_size() == 0`` indicates no sub blocks.
  for (auto& attr : op_desc.attrs()) {
    if (attr.type() == proto::AttrType::BLOCKS) {
      PADDLE_ENFORCE_GT(
          attr.blocks_idx_size(),
          0,
          common::errors::NotFound(
              "Attribute blocks is not found in operator %s", op_desc.type()));
      return true;
    }
  }

  return false;
}

int GetOpRole(const proto::OpDesc& op_desc) {
  for (auto& attr : op_desc.attrs()) {
    if (attr.name() == OpProtoAndCheckerMaker::OpRoleAttrName()) {
      PADDLE_ENFORCE_EQ(
          attr.has_i(),
          true,
          common::errors::NotFound("Attribute %s is empty in operator %s",
                                   OpProtoAndCheckerMaker::OpRoleAttrName(),
                                   op_desc.type()));
      return attr.i();
    }
  }
  // If attr op_role is not found, it may be operator created in c++ test, like
  // prune_test.cc. In that case, the op_role should be default value, which is
  // kNotSpecified.
  return static_cast<int>(OpRole::kNotSpecified);
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

int FindMapByValue(const std::map<int, int>& m, int val) {
  // The content in map should be >= 0, so -1 is used to indicate "NotFound".
  for (auto& pair : m) {
    if (pair.second == val) {
      return pair.first;
    }
  }
  return -1;
}

// In other two cases, the op that has feed vars as output vars is dependent:
// 1. op has subblock, like while/for/ifelse/recurrent/pylayer
// 2. op is in subblock
bool IsSubBlockDependent(const proto::OpDesc& op_desc,
                         const std::set<std::string>& feed_vars,
                         int parent_block_id) {
  for (auto& var : op_desc.outputs()) {
    for (auto& argu : var.arguments()) {
      if ((HasSubBlock(op_desc) || HasSubBlocks(op_desc) ||
           parent_block_id != -1) &&
          feed_vars.count(argu) != 0) {
        return true;
      }
    }
  }
  return false;
}

// block_id is the idx of the current block in the input desc
// parent_block_id is the idx of the parent of the current block
// in the output desc, -1 means the current block is global block
// dependent_vars is passed recursively from the parent block to
// the child block to help pruning
void prune_impl(const proto::ProgramDesc& input,
                proto::ProgramDesc* output,
                int block_id,
                int parent_block_id,
                std::unordered_set<std::string>* dependent_vars,
                const std::set<std::string> feed_var_names,
                std::map<int, int>* pruned_origin_block_id_map) {
  auto& block = input.blocks(block_id);
  auto& ops = block.ops();
  auto add_dependent_var = [&](const std::string& name) {
    if (feed_var_names.count(name) == 0) dependent_vars->insert(name);
  };

  bool expect_feed = true;
  for (auto& op_desc : ops) {
    PADDLE_ENFORCE_EQ(
        op_desc.type() != kFeedOpType || expect_feed,
        true,
        common::errors::PreconditionNotMet(
            "All FeedOps are at the beginning of the ProgramDesc"));
    expect_feed = (op_desc.type() == kFeedOpType);
  }

  bool expect_fetch = true;
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    PADDLE_ENFORCE_EQ(op_desc.type() != kFetchOpType || expect_fetch,
                      true,
                      common::errors::PreconditionNotMet(
                          "All FetchOps must at the end of the ProgramDesc"));
    expect_fetch = (op_desc.type() == kFetchOpType);
  }

  std::vector<bool> should_run;
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;

    // TODO(wanghaipeng03) reconstruct the following if/else block
    //                     to extract common code
    //
    // bool should_run_flag = false;
    // if (IsTarget........) {
    //   should_run_flag = true;
    // } else {
    //   if (parent......) {
    //     for (....) {
    //       for (.....) {
    //         if (.....) {
    //           should_run_flag = true;
    //         }
    //       }
    //     }
    //   }
    // }
    //
    // should_run.push_back(should_run_flag);
    // if (should_run_flag) {
    //   for (auto & var: op_desc.inputs()) {
    //     for (....) {
    //       if (.....) {
    //         dependent_vars->insert(argu);
    //       }
    //     }
    //   }
    // }

    if (IsTarget(op_desc) ||
        ((HasDependentOutputVar(op_desc, *dependent_vars) ||
          (IsSubBlockDependent(op_desc, feed_var_names, parent_block_id))) &&
         (GetOpRole(op_desc) & static_cast<int>(OpRole::kOptimize)) == 0)) {
      // NOTE(zhiqiu): since optimize op takes the trainable parameters as
      // inputs and output, it may introduce wrong dependency graph.
      // For train mode, the optimize op should be in targets, so is not need
      // and not right to mark optimize op by its outputs.
      // For eval / infer mode, there is no optimize op in program.
      for (auto& var : op_desc.inputs()) {
        for (auto& argu : var.arguments()) {
          add_dependent_var(argu);
        }
      }
      // NOTE(dev): All attribute with VarDesc type is considered as Input,
      // so they shall be added into dependent_vars.
      for (auto& attr : op_desc.attrs()) {
        if (attr.type() == proto::AttrType::VAR) {
          add_dependent_var(attr.var_name());
        } else if (attr.type() == proto::AttrType::VARS) {
          for (auto& name : attr.vars_name()) {
            add_dependent_var(name);
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

  auto* op_field = output_block->mutable_ops();
  op_field->Clear();
  for (size_t i = 0; i < should_run.size(); ++i) {
    if (should_run[i]) {
      auto* op = op_field->Add();
      *op = input.blocks(block_id).ops(static_cast<int>(i));
      if (HasSubBlock(*op) || HasSubBlocks(*op)) {
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
        if (HasSubBlock(*op)) {
          // GetSubBlockIndex(*op) is the idx of the sub_block in the input desc
          // output_block_id is the idx of the current block in the output desc
          prune_impl(input,
                     output,
                     GetSubBlockIndex(*op),
                     output_block_id,
                     &sub_block_dependent_vars,
                     feed_var_names,
                     pruned_origin_block_id_map);
        } else if (HasSubBlocks(*op)) {
          // GetSubBlocksIndices(*op) are the indices of the sub_blocks in the
          // input desc output_block_id is the idx of the current block in the
          // output desc
          std::vector<int> sub_indices;
          GetSubBlocksIndices(*op, &sub_indices);
          for (auto& sub_index : sub_indices) {
            // create a copy of dependent_vars to avoid being overwritten by the
            // other sub_block
            std::unordered_set<std::string> dependent_vars_copy =
                sub_block_dependent_vars;
            prune_impl(input,
                       output,
                       sub_index,
                       output_block_id,
                       &dependent_vars_copy,
                       feed_var_names,
                       pruned_origin_block_id_map);
          }
        } else {
          PADDLE_ENFORCE(false,
                         common::errors::PreconditionNotMet(
                             "Attr Block or Blocks must exist when recursively "
                             "calling prune_impl"));
        }
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
  auto add_var_names = [&](const std::string& name) {
    if (var_map.count(name) != 0) var_names.insert(name);
  };
  for (const auto& op : *op_field) {
    auto& input_field = op.inputs();
    for (auto& input_var : input_field) {
      for (auto& arg : input_var.arguments()) {
        add_var_names(arg);
      }
    }
    auto& output_field = op.outputs();
    for (auto& output_var : output_field) {
      for (auto& arg : output_var.arguments()) {
        add_var_names(arg);
      }
    }
    // NOTE(dev): All attribute with VarDesc type is considered as Input,
    // so they shall be added into dependent_vars.
    for (auto& attr : op.attrs()) {
      if (attr.type() == proto::AttrType::VAR) {
        add_var_names(attr.var_name());
      } else if (attr.type() == proto::AttrType::VARS) {
        for (auto& name : attr.vars_name()) {
          add_var_names(name);
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
  prune_impl(input,
             output,
             0,
             -1,
             &dependent_vars,
             feed_var_names,
             &pruned_origin_block_id_map);
  // update subblock idx
  for (int i = 0; i < output->blocks_size(); i++) {
    auto* pruned = output->mutable_blocks(i);
    auto* ops = pruned->mutable_ops();
    for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
      auto& op_desc = *op_iter;
      if (HasSubBlock(op_desc)) {
        int origin_sub_idx = GetSubBlockIndex(op_desc);
        auto sub_idx =
            FindMapByValue(pruned_origin_block_id_map, origin_sub_idx);
        PADDLE_ENFORCE_NE(
            sub_idx,
            -1,
            common::errors::NotFound(
                "The origin sub block id should be found in "
                "pruned_progin_block_id_map when the op has sub_block"));
        SetSubBlockIndex(&op_desc, sub_idx);
      } else if (HasSubBlocks(op_desc)) {
        std::vector<int> origin_sub_indices;
        GetSubBlocksIndices(op_desc, &origin_sub_indices);
        std::vector<int> sub_indices;
        for (int index : origin_sub_indices) {
          auto sub_idx = FindMapByValue(pruned_origin_block_id_map, index);
          PADDLE_ENFORCE_NE(
              sub_idx,
              -1,
              common::errors::NotFound(
                  "The origin sub block id should be found in "
                  "pruned_progin_block_id_map when the op has sub_blocks"));
          sub_indices.push_back(sub_idx);
        }

        SetSubBlocksIndices(&op_desc, sub_indices);
      }
    }
  }
  return pruned_origin_block_id_map;
}

void PruneBackwardImpl(proto::BlockDesc* origin, proto::BlockDesc* pruned) {
  std::unordered_set<std::string> op_input_vars;
  std::unordered_set<std::string> op_output_vars;

  // Step 1. Mark backward, optimize and lrsched ops in the block
  auto* ops = origin->mutable_ops();
  for (auto& op_desc : *ops) {
    auto op_role = GetOpRole(op_desc);
    if (op_role & static_cast<int>(OpRole::kOptimize) ||
        op_role & static_cast<int>(OpRole::kBackward) ||
        op_role & static_cast<int>(OpRole::kLRSched)) {
      op_desc.set_is_target(false);
    }
  }

  // Step 2. Copy the forward ops which have not been set false target to new
  // ProgramDesc
  // Note: The proto::ProgramDesc doesn't have interface
  //       to remove op and var
  auto* op_field = pruned->mutable_ops();
  op_field->Clear();
  for (auto& op_desc : *ops) {
    if (!HasFalseTarget(op_desc)) {
      auto* op = op_field->Add();
      AppendOpInputVarNames(op_desc, &op_input_vars);
      AppendOpOutputVarNames(op_desc, &op_output_vars);
      *op = op_desc;

      // if the type of op is "pylayer", we need to update the ``blocks``
      // attribute because the backward block will be pruned
      if (op->type() == kPyLayer && HasSubBlocks(*op)) {
        std::vector<int> sub_indices;
        GetSubBlocksIndices(*op, &sub_indices);
        if (sub_indices.size() > 1) {
          // sub_indices contains both forward block id and backward block id
          std::vector<int> new_sub_indices(sub_indices.begin(),
                                           sub_indices.end() - 1);
          SetSubBlocksIndices(op, new_sub_indices);
        }
      }
    }
  }

  // Step 3. Copy the forward vars to new ProgramDesc,
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
    if (var_map.count(name)) {
      // NOTE(zhiqiu): For operator in a conditional block, the related vars
      // may not exist in current block, but in its further block.
      *pruned_vars->Add() = var_map[name];
    }
  }
}  // namespace framework

std::tuple<framework::ProgramDesc, std::map<int, int>> PruneBackward(
    const framework::ProgramDesc& origin) {
  // Copy original ProgramDesc, origin can't be change
  framework::ProgramDesc origin_clone(origin);

  // Step 1. check if the program contains grad loss operator or pylayer
  // operator. If not, the program need no pruning.
  bool has_loss_grad_op = false;
  bool has_pylayer_op = false;
  std::queue<int> block_contains_loss;
  std::queue<int> block_contains_loss_grad;
  for (size_t i = 0; i < origin_clone.Size(); i++) {
    auto block_ops = origin_clone.Block(i).AllOps();
    for (auto op : block_ops) {
      int op_role = PADDLE_GET_MUTABLE(
          int, op->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
      if (op_role == (static_cast<int>(OpRole::kBackward) |
                      static_cast<int>(OpRole::kLoss))) {
        op->SetIsTarget(false);
        has_loss_grad_op = true;
      }
      if (op->Type() == kPyLayer) {
        has_pylayer_op = true;
      }
    }
  }

  std::map<int, int> pruned_progin_block_id_map;
  if (!has_loss_grad_op && !has_pylayer_op) {
    // No pruning, fast return a copy of the origin ProgramDesc with an empty
    // map, means default mapped, i.e.{0:0, 1:1, ..., n:n}.
    return std::make_tuple(framework::ProgramDesc(origin_clone),
                           pruned_progin_block_id_map);
  }

  proto::ProgramDesc pruned_desc;
  pruned_desc.clear_blocks();

  // Step 2. Prune backward for each block.
  for (size_t i = 0; i < origin_clone.Size(); i++) {
    auto pruned = proto::BlockDesc();
    auto origin = origin_clone.Proto()->mutable_blocks(static_cast<int>(i));

    PruneBackwardImpl(origin, &pruned);
    // If pruned block contains no operator, it means the block is a
    // backward block and should be pruned.
    // Else, add the block to pruned_desc and update its id & parent_id.
    if (pruned.ops_size() > 0) {
      auto* block_field = pruned_desc.mutable_blocks();
      *block_field->Add() = pruned;

      auto pruned_block_id = pruned_desc.blocks_size() - 1;
      pruned_progin_block_id_map[pruned_block_id] = origin->idx();
      auto* pruned_block = pruned_desc.mutable_blocks(pruned_block_id);
      pruned_block->set_idx(pruned_block_id);

      if (origin->parent_idx() == -1) {
        pruned_block->set_parent_idx(-1);
      } else {
        auto parent_idx =
            FindMapByValue(pruned_progin_block_id_map, origin->parent_idx());
        PADDLE_ENFORCE_NE(parent_idx,
                          -1,
                          common::errors::NotFound(
                              "The origin parent block id is not found in "
                              "pruned_progin_block_id_map"));
        pruned_block->set_parent_idx(parent_idx);
      }
    }
  }

  // Step 3. Update subblock attribute for conditional operator.
  // This should be performed after all blocks pruned.
  for (int i = 0; i < pruned_desc.blocks_size(); i++) {
    auto* pruned = pruned_desc.mutable_blocks(i);
    auto* ops = pruned->mutable_ops();
    for (auto& op_desc : *ops) {
      if (HasSubBlock(op_desc)) {
        int origin_sub_idx = GetSubBlockIndex(op_desc);
        auto sub_idx =
            FindMapByValue(pruned_progin_block_id_map, origin_sub_idx);
        PADDLE_ENFORCE_NE(
            sub_idx,
            -1,
            common::errors::NotFound(
                "The origin sub block id is not found in "
                "pruned_progin_block_id_map when the op has sub_block"));
        SetSubBlockIndex(&op_desc, sub_idx);
      } else if (HasSubBlocks(op_desc)) {
        std::vector<int> origin_sub_indices;
        GetSubBlocksIndices(op_desc, &origin_sub_indices);
        std::vector<int> sub_indices;
        for (int index : origin_sub_indices) {
          auto sub_idx = FindMapByValue(pruned_progin_block_id_map, index);
          PADDLE_ENFORCE_NE(
              sub_idx,
              -1,
              common::errors::NotFound(
                  "The origin sub block id should be found in "
                  "pruned_progin_block_id_map when the op has sub_blocks"));
          sub_indices.push_back(sub_idx);
        }

        SetSubBlocksIndices(&op_desc, sub_indices);
      }
    }
  }

  // Step 4. Return a tuple
  return std::make_tuple(framework::ProgramDesc(pruned_desc),
                         pruned_progin_block_id_map);
}

}  // namespace paddle::framework
