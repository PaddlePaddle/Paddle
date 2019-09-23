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

#include "paddle/fluid/framework/details/share_tensor_buffer_functor.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {

// TODO(zjl): support SelectedRows
static inline const Tensor &GetTensorFromVar(const Variable *var) {
  if (var->IsType<LoDTensor>()) {
    return var->Get<LoDTensor>();
  } else {
    PADDLE_THROW("Variable must be type of LoDTensor");
  }
}

static inline Tensor *GetMutableTensorFromVar(Variable *var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else {
    PADDLE_THROW("Variable must be type of LoDTensor");
  }
}

ShareTensorBufferFunctor::ShareTensorBufferFunctor(
    Scope *scope, size_t scope_idx, const std::string &op_type,
    const std::vector<const ir::MemOptVarInfo *> &in_var_infos,
    const std::vector<std::string> &out_var_names)
    : scope_(scope),
      scope_idx_(scope_idx),
      op_type_(op_type),
      in_var_infos_(in_var_infos),
      out_var_names_(out_var_names) {
  PADDLE_ENFORCE_EQ(in_var_infos_.size(), out_var_names_.size());
  for (size_t i = 0; i < in_var_infos_.size(); ++i) {
    AddReuseVarPair(in_var_infos_[i], out_var_names_[i]);
  }
}

std::unordered_map<std::string, std::string>
ShareTensorBufferFunctor::ReusedVars() const {
  std::unordered_map<std::string, std::string> result;
  for (size_t i = 0; i < in_var_infos_.size(); ++i) {
    result.insert({in_var_infos_[i]->Name(), out_var_names_[i]});
  }
  return result;
}

void ShareTensorBufferFunctor::AddReuseVarPair(
    const ir::MemOptVarInfo *in_var_info, const std::string &out_var_name) {
  PADDLE_ENFORCE_NOT_NULL(in_var_info, "in_var_info cannot be nullptr");
  PADDLE_ENFORCE_NE(in_var_info->Name(), out_var_name,
                    "in/out cannot have same name: %s", out_var_name);
  in_var_infos_.emplace_back(in_var_info);
  out_var_names_.emplace_back(out_var_name);
}

void ShareTensorBufferFunctor::CallOnce() {
  PADDLE_ENFORCE(in_out_vars_.empty(), "in_out_vars_ must be initialized here");
  for (size_t i = 0; i < in_var_infos_.size(); ++i) {
    auto *in_var = exec_scope_->FindVar(in_var_infos_[i]->Name());
    auto *out_var = exec_scope_->FindVar(out_var_names_[i]);
    PADDLE_ENFORCE_NOT_NULL(in_var);
    PADDLE_ENFORCE_NOT_NULL(out_var);
    PADDLE_ENFORCE_NE(in_var, out_var);
    in_out_vars_.emplace_back(in_var, out_var);
  }
}

void ShareTensorBufferFunctor::operator()(Scope *exec_scope) {
  if (!exec_scope_) {
    PADDLE_ENFORCE_NOT_NULL(exec_scope);
    exec_scope_ = exec_scope;
    CallOnce();
  } else {
    PADDLE_ENFORCE(exec_scope_ == exec_scope, "Scope must be the same");
  }

  for (size_t i = 0; i < in_var_infos_.size(); ++i) {
    const auto &in_tensor = GetTensorFromVar(in_out_vars_[i].first);
    auto *out_tensor = GetMutableTensorFromVar(in_out_vars_[i].second);
    auto *in_var_info = in_var_infos_[i];

    if (UNLIKELY(in_var_info->IsSkippedMemoryReuse())) {
      // If in_var is inplaced in the previous batch and we want to fetch
      // in_var in the current batch, we have to reset memory of out_var
      // to avoid wrong calculation result.
      if (in_tensor.Holder() == out_tensor->Holder()) {
        VLOG(1) << "Clear " << out_var_names_[i]
                << " because you may want to fetch an inplaced variable "
                << in_var_info->Name()
                << " in previous batch: " << in_var_info->Name() << " -> "
                << out_var_names_[i];
        out_tensor->clear();
      }
    } else {
      out_tensor->ShareBufferWith(in_tensor);

      VLOG(2) << "Share tensor buffer when running " << op_type_ << " : "
              << in_var_info->Name() << " -> " << out_var_names_[i];
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
