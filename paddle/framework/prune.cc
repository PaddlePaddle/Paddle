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
#include <vector>

#include <glog/logging.h>

namespace paddle {
namespace framework {

const std::string kFeedOpType = "feed";
const std::string kFetchOpType = "fetch";

bool HasDependentVar(const OpDesc& op_desc,
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

bool IsTarget(const OpDesc& op_desc) {
  if (op_desc.has_is_target()) {
    return op_desc.is_target();
  }
  return false;
}

void prune_impl(const ProgramDesc& input, ProgramDesc* output, int block_id) {
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
}

// TODO(fengjiayi): Prune() could be inplaced to avoid unnecessary copies
void Prune(const ProgramDesc& input, ProgramDesc* output) {
  prune_impl(input, output, 0);
}

}  // namespace framework
}  // namespace paddle
