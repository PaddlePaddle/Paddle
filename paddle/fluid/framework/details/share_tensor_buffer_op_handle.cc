// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace framework {
namespace details {

ShareTensorBufferOpHandle::ShareTensorBufferOpHandle(
    ir::Node *node, const Scope *scope, size_t scope_idx,
    const std::vector<ir::MemOptVarInfo *> &in_vars,
    const std::vector<std::string> &out_vars)
    : OpHandleBase(node),
      scope_(scope),
      scope_idx_(scope_idx),
      in_vars_(in_vars),
      out_vars_(out_vars),
      is_shared_(in_vars.size(), false) {
  PADDLE_ENFORCE(!in_vars_.empty(), "in_vars_ cannot be empty");
  for (auto &in_var : in_vars_) {
    PADDLE_ENFORCE_NOT_NULL(in_var, "in_var cannot be nullptr");
  }
  PADDLE_ENFORCE_EQ(in_vars_.size(), out_vars_.size());
}

std::unordered_set<std::string> ShareTensorBufferOpHandle::ReusedVarSet()
    const {
  std::unordered_set<std::string> result;
  for (auto &in_var : in_vars_) {
    result.insert(in_var->Name());
  }
  return result;
}

Tensor *ShareTensorBufferOpHandle::GetTensor(Scope **exec_scope,
                                             const std::string &name) {
  if (*exec_scope == nullptr) {
    *exec_scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();
  }

  auto *var = (*exec_scope)->FindVar(name);
  if (var == nullptr) {
    return nullptr;
  }

  // TODO(zjl): to support SelectedRows
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else {
    return nullptr;
  }
}

void ShareTensorBufferOpHandle::RunImpl() {
  Scope *exec_scope = nullptr;
  for (size_t i = 0; i < in_vars_.size(); ++i) {
    auto in_var = in_vars_[i];
    if (in_var->IsSkipped()) {
      VLOG(1) << MemoryReuseDebugString(i)
              << " is disabled, because we want to fetch " << in_var->Name();
      // If in_var is inplaced in the previous batch and we want to fetch
      // in_var in the current batch, we have to reset memory of out_var
      // to avoid wrong calcualtion result.
      if (is_shared_[i]) {
        // You have to reset memory here to avoid caculation wrong!
        VLOG(1) << MemoryReuseDebugString(i)
                << " is performed in previous batch, "
                << "we have to reset " << out_vars_[i];

        // Clear out_tensor because this tensor may be shared in the previous
        // batch
        auto *out_tensor = GetTensor(&exec_scope, out_vars_[i]);
        if (out_tensor) {
          out_tensor->clear();
        }

        is_shared_[i] = false;
      }
      continue;
    }

    is_shared_[i] = false;

    auto *in_tensor = GetTensor(&exec_scope, in_var->Name());
    if (!in_tensor) {
      continue;
    }

    auto *out_tensor = GetTensor(&exec_scope, out_vars_[i]);
    if (!out_tensor) {
      continue;
    }

    VLOG(2) << "Perform " << MemoryReuseDebugString(i);

    out_tensor->ShareBufferWith(*in_tensor);

    is_shared_[i] = true;
  }
}

InplaceShareTensorBufferOpHandle::InplaceShareTensorBufferOpHandle(
    ir::Node *node, const Scope *scope, size_t scope_idx,
    const std::string &op_type,
    const std::vector<std::pair<std::string, std::string>> &in_out_params,
    const std::vector<ir::MemOptVarInfo *> &in_vars,
    const std::vector<std::string> &out_vars)
    : ShareTensorBufferOpHandle(node, scope, scope_idx, in_vars, out_vars),
      op_type_(op_type),
      in_out_params_(in_out_params) {
  PADDLE_ENFORCE_EQ(in_out_params_.size(), in_vars_.size());
}

std::string InplaceShareTensorBufferOpHandle::MemoryReuseDebugString(
    size_t i) const {
  return "\"Inplace " + op_type_ + ": " + in_out_params_[i].first + "(" +
         in_vars_[i]->Name() + ") -> " + in_out_params_[i].second + "(" +
         out_vars_[i] + ")\"";
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
