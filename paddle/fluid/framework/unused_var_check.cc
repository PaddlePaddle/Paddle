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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/unused_var_check.h"
#include "paddle/fluid/platform/enforce.h"

DEFINE_bool(enable_unused_var_check, true,
            "Checking whether operator contains unused inputs, "
            "especially for grad operator. It should be in unittest.");

namespace paddle {
namespace framework {

std::unordered_set<std::string> *GetThreadLocalUsedVarNameSet() {
  thread_local std::unordered_set<std::string> used_var_name_set;
  return &used_var_name_set;
}

void CheckUnusedVar(const OperatorBase &op, const Scope &scope) {
  // skip op in white list
  if (op_has_unsed_vars_white_list.count(op.Type()) != 0) {
    return;
  }
  auto *used_set = GetThreadLocalUsedVarNameSet();
  std::vector<std::string> unsed_input_var_names;
  auto &inferer = op.Info().NoNeedBufferVarsInferer();
  std::unordered_set<std::string> no_need_buffer_ins = {};
  if (inferer) {
    no_need_buffer_ins = inferer(op.Inputs(), op.Outputs(), op.Attrs());
  }

  for (auto &pair : op.Inputs()) {
    // skip no need buffer vars declared
    if (no_need_buffer_ins.count(pair.first) != 0) {
      VLOG(6) << op.Type() << " " << pair.first;
      continue;
    }
    if (used_set->count(pair.first) == 0) {
      for (auto &in_var_name : pair.second) {
        auto *in_var = scope.FindVar(in_var_name);
        auto &tensor = in_var->Get<LoDTensor>();
        if (in_var->IsInitialized() && tensor.IsInitialized()) {
          unsed_input_var_names.emplace_back(pair.first);
          break;
        }
      }
    }
  }
  if (!unsed_input_var_names.empty()) {
    std::string err_msg = "Operator " + op.Type() + " has input(s) not uesed: ";
    for (auto &in_var_name : unsed_input_var_names) {
      err_msg += in_var_name;
      err_msg += ", ";
    }
    err_msg += "please remove it from inputs or register NoNeedBufferVars!";
    VLOG(1) << err_msg;
  }
}

}  // namespace framework
}  // namespace paddle
