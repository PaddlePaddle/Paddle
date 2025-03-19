// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/details/reset_tensor_array.h"

#include "glog/logging.h"

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::details {

// Should be called after the parameters are loaded.
void TensorArrayBatchCleaner::CollectTensorArrays(framework::Scope *scope) {
  if (flag_) {
    for (auto &var_name : scope->LocalVarNames()) {
      auto *var = scope->FindVar(var_name);
      // TODO(Superjomn) should avoid the case when a TensorArray is a
      // parameter.
      if (var_name == "feed" || var_name == "fetch") continue;
      if (var->IsType<phi::TensorArray>()) {
        VLOG(4) << "collect " << var_name;
        arrays_.push_back(var->GetMutable<phi::TensorArray>());
      }
    }
    for (auto *kid : scope->kids()) {
      CollectTensorArrays(kid);
    }

    VLOG(3) << "Collect " << arrays_.size() << " arrays";
    flag_ = false;
  }
}

// Should be called when `Run` finished.
void TensorArrayBatchCleaner::ResetTensorArray() {
  for (auto *arr : arrays_) {
    arr->clear();
  }
}

void TensorArrayBatchCleaner::CollectNoTensorVars(framework::Scope *scope) {
  if (no_tensor_flag_) {
    for (auto &var_name : scope->LocalVarNames()) {
      auto *var = scope->FindVar(var_name);
      if (!var->IsInitialized()) continue;
      if (!valid_types_.count(var->Type())) {
        no_tensor_vars_.insert(var);
      }
    }

    for (auto *kid : scope->kids()) {
      CollectTensorArrays(kid);
    }
    no_tensor_flag_ = false;  // Only collect one time.
  }
}

void TensorArrayBatchCleaner::ResetNoTensorVars() {
  for (auto *var : no_tensor_vars_) {
    var->Clear();
  }
}

}  // namespace paddle::details
