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

DEFINE_bool(enable_unused_var_check, false,
            "Checking whether operator contains unused inputs, "
            "especially for grad operator. It should be in unittest.");

const std::unordered_set<std::string> op_has_unsed_vars_white_list = {
    "auc",
    "batch_norm",
    "batch_norm_grad",
    "fused_batch_norm_act",
    "fused_batch_norm_act_grad",
    "sync_batch_norm_grad",
    "center_loss_grad",
    "crop",
    "cvm",
    "cos_sim_grad",
    "dgc_momentum",
    "fake_quantize_range_abs_max",
    "fill_zeros_like",
    "fusion_seqpool_cvm_concat",
    "reshape2_grad_grad",
    "reshape2_grad",
    "gru_grad",
    "hierarchical_sigmoid_grad",
    "nce_grad",
    "roi_perspective_transform_grad",
    "sequence_conv_grad",
    "gru_unit_grad",
    "affine_grid_grad",
    "fill_any_like",
    "precision_recall",
    "unsqueeze_grad",
    "kldiv_loss_grad",
    "cvm_grad",
    "stack_grad",
    "warpctc_grad",
    "sync_batch_norm",
    "match_matrix_tensor_grad",
    "ngraph_engine",
    "rmsprop"};

namespace paddle {
namespace framework {

std::unordered_set<std::string> *GetThreadLocalUsedVarNameSet() {
  thread_local std::unordered_set<std::string> used_var_name_set;
  return &used_var_name_set;
}

void LogVarUsageIfUnusedVarCheckEnabled(const std::string &name) {
  if (FLAGS_enable_unused_var_check) {
    VLOG(6) << "Variable used:" << name;
    GetThreadLocalUsedVarNameSet()->insert(name);
  }
}

void CheckUnusedVar(const OperatorBase &op, const Scope &scope) {
  // skip op in white list and it should be fixed in the future.
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
        if (in_var != nullptr && in_var->IsInitialized()) {
          auto *tensor = &in_var->Get<LoDTensor>();
          if (tensor != nullptr && tensor->IsInitialized()) {
            unsed_input_var_names.emplace_back(pair.first);
            break;
          }
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
    err_msg +=
        "please make sure it(they) is(are) needed. If not, remove it(them) "
        "from inputs of the operator; if yes, register "
        "NoNeedBufferVarsInference or add "
        "the operator to "
        "white list in unused_var_check.cc. See more details at "
        "[https://github.com/PaddlePaddle/Paddle/wiki/"
        "OP-Should-Not-Have-Unused-Input]";
    PADDLE_ENFORCE_EQ(unsed_input_var_names.size(), 0,
                      platform::errors::PermissionDenied(
                          "Unused input variables check failed: %s", err_msg));
  }
}

}  // namespace framework
}  // namespace paddle
