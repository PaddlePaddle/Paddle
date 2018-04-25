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
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {

const char kFeedOpType[] = "feed";
const char kFetchOpType[] = "fetch";

bool HasDependentVar(const proto::OpDesc& op_desc,
                     const std::set<std::string>& dependent_vars) {
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

// block_id is the idx of the current block in the input desc
// parent_block_id is the idx of the parent of the current block
// in the output desc, -1 means the current block is global block
// dependent_vars is passed recursively from the parent block to
// the child block to help pruning
void prune_impl(const proto::ProgramDesc& input, proto::ProgramDesc* output,
                int block_id, int parent_block_id,
                std::set<std::string>* dependent_vars) {
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
    if (IsTarget(op_desc) || HasDependentVar(op_desc, *dependent_vars)) {
      // insert its input to the dependency graph
      for (auto& var : op_desc.inputs()) {
        for (auto& argu : var.arguments()) {
          dependent_vars->insert(argu);
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
        // create sub_block_dependent_vars here to help prune the sub block
        std::set<std::string> sub_block_dependent_vars;
        for (auto& var : op->inputs()) {
          for (auto& argu : var.arguments()) {
            sub_block_dependent_vars.insert(argu);
          }
        }
        for (auto& var : op->outputs()) {
          for (auto& argu : var.arguments()) {
            sub_block_dependent_vars.insert(argu);
          }
        }
        // GetSubBlockIndex(*op) is the idx of the sub_block in the input desc
        // output_block_id is the idx of the current block in the output desc
        prune_impl(input, output, GetSubBlockIndex(*op), output_block_id,
                   &sub_block_dependent_vars);
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
void Prune(const proto::ProgramDesc& input, proto::ProgramDesc* output) {
  std::set<std::string> dependent_vars;
  output->clear_blocks();
  prune_impl(input, output, 0, -1, &dependent_vars);
}

void inference_optimize_impl(proto::ProgramDesc* input, int block_id) {
  auto* op_field = input->mutable_blocks(block_id)->mutable_ops();
  for (auto& op_desc : *op_field) {
    for (auto& attr : *op_desc.mutable_attrs()) {
      if (attr.name() == "is_test") {
        attr.set_b(true);
        break;
      }
    }
  }
}

void InferenceOptimize(const proto::ProgramDesc& input,
                       proto::ProgramDesc* output) {
  *output = input;
  int num_blocks = output->blocks_size();
  PADDLE_ENFORCE_GT(num_blocks, 0, "ProgramDesc must have at least one block");
  for (int i = 0; i < num_blocks; ++i) {
    inference_optimize_impl(output, i);
  }
}

}  // namespace framework
}  // namespace paddle
