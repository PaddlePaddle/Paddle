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

namespace paddle {
namespace details {

// Should be called after the parameters are loaded.
void TensorArrayBatchCleaner::CollectTensorArrays(framework::Scope *scope) {
  if (flag_) {
    for (auto &var_name : scope->LocalVarNames()) {
      auto *var = scope->FindVar(var_name);
      // TODO(Superjomn) should avoid the case when a TensorArray is a
      // parameter.
      if (var_name == "feed" || var_name == "fetch") continue;
      if (var->Type() == typeid(framework::LoDTensorArray) ||
          var->Type() == typeid(framework::Scope *) ||
          var->Type() == typeid(std::vector<int32_t>) ||
          var->Type() == typeid(std::vector<int64_t>) ||
          var->Type() == typeid(std::vector<float>) ||
          var->Type() == typeid(std::vector<double>) ||
          var->Type() == typeid(std::vector<std::string>)) {
        VLOG(4) << "collect " << var_name;
        arrays_.insert(var->GetMutable<framework::LoDTensorArray>());
      }
    }
    for (auto *kid : scope->kids()) {
      CollectTensorArrays(kid);
    }

    LOG(INFO) << "Collect " << arrays_.size() << " arrays";
    flag_ = true;
  }
}

// Should be called when `Run` finished.
void TensorArrayBatchCleaner::ResetTensorArray() {
  for (auto *arr : arrays_) {
    arr->clear();
  }
}

void TensorArrayBatchCleaner::CollectOtherTypes(framework::Scope *scope) {
  if (!no_tensor_flag_) {
    for (auto& var_name : scope->LocalVarNames()) {
      auto* var = scope->FindVar(var_name);
      if (!valid_types_.count(var->Type())) {
        no_tensor_vars_.insert(var);
      }
    }
    LOG(INFO) << "collect " << no_tensor_vars_.size() << " no tensor vars";
    no_tensor_flag_ = false;
  }
}

void TensorArrayBatchCleaner::ResetOtherTypes() {
  for (auto* var : no_tensor_vars_) {
    // Clear the original data structure.
    var->GetMutable<char>();
  }
}

}  // namespace details
}  // namespace paddle
