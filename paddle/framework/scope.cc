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

#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

Variable* Scope::CreateVariable(const std::string& name) {
  if (!HasVariable(name)) {
    vars_[name] = std::unique_ptr<Variable>(new Variable());
  }
  return GetVariable(name);
}

Variable* Scope::GetVarLocally(const std::string& name) const {
  if (vars_.count(name)) {
    return vars_.at(name).get();
  }
  return nullptr;
}

Variable* Scope::GetVariable(const std::string& name) const {
  Variable* var = GetVarLocally(name);
  if (var != nullptr) {
    return var;
  } else if (parent_ != nullptr) {
    return parent_->GetVariable(name);
  } else {
    return nullptr;
  }
}

bool Scope::HasVariable(const std::string &name) {
  return (vars_.count(name) > 0 || (parent_ && parent_->HasVariable(name)));
}

}  // namespace framework
}  // namespace paddle
