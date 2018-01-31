/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/prune.h"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>

namespace paddle {
namespace framework {

const std::string kFeedOpType = "feed";
const std::string kFetchOpType = "fetch";
const std::string kDropOutOpType = "dropout";
const std::string kBatchNormOpType = "batch_norm";

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

void prune_impl(const proto::ProgramDesc& input, proto::ProgramDesc* output,
                int block_id) {
  // TODO(tonyyang-svail):
  //    - will change to use multiple blocks for RNN op and Cond Op

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

  std::set<std::string> dependent_vars;
  std::vector<bool> should_run;
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;

    if (IsTarget(op_desc) || HasDependentVar(op_desc, dependent_vars)) {
      // insert its input to the dependency graph
      for (auto& var : op_desc.inputs()) {
        for (auto& argu : var.arguments()) {
          dependent_vars.insert(argu);
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

  *output = input;
  auto* op_field = output->mutable_blocks(block_id)->mutable_ops();
  op_field->Clear();
  for (size_t i = 0; i < should_run.size(); ++i) {
    if (should_run[i]) {
      *op_field->Add() = input.blocks(block_id).ops(i);
    }
  }

  // remove the VarDescs in BlockDesc that are not referenced in
  // the pruned OpDescs
  std::unordered_map<std::string, proto::VarDesc> var_map;
  auto* var_field = output->mutable_blocks(block_id)->mutable_vars();
  for (const auto& var : *var_field) {
    var_map[var.name()] = var;
  }

  var_field->Clear();
  for (const auto& op : *op_field) {
    // add VarDescs of all input arguments for each OpDesc
    auto& input_field = op.inputs();
    for (auto& input_var : input_field) {
      for (auto& arg : input_var.arguments()) {
        *var_field->Add() = var_map[arg];
      }
    }
    // add VarDescs of all output arguments for each OpDesc
    auto& output_field = op.outputs();
    for (auto& output_var : output_field) {
      for (auto& arg : output_var.arguments()) {
        *var_field->Add() = var_map[arg];
      }
    }
  }
}

// TODO(fengjiayi): Prune() could be inplaced to avoid unnecessary copies
void Prune(const proto::ProgramDesc& input, proto::ProgramDesc* output) {
  prune_impl(input, output, 0);
}

void inference_optimize_impl(const proto::ProgramDesc& input,
                             proto::ProgramDesc* output, int block_id) {
  *output = input;
  auto* op_field = output->mutable_blocks(block_id)->mutable_ops();
  for (auto& op_desc : *op_field) {
    if (op_desc.type() == kDropOutOpType ||
        op_desc.type() == kBatchNormOpType) {
      for (auto& attr : *op_desc.mutable_attrs()) {
        if (attr.name() == "is_test") {
          attr.set_b(true);
          break;
        }
      }
    }
  }
}

void InferenceOptimize(const proto::ProgramDesc& input,
                       proto::ProgramDesc* output) {
  inference_optimize_impl(input, output, 0);
}

}  // namespace framework
}  // namespace paddle
