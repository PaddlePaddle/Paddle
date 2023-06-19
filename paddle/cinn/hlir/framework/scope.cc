// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/scope.h"

#include "paddle/cinn/common/common.h"

namespace cinn {
namespace hlir {
namespace framework {

void Scope::EraseVar(const std::string& name) {
  CHECK(data_.count(name)) << "Variable(" << name << ") not found";
  data_.erase(name);
}

Variable* Scope::FindVar(const std::string& name) const {
  auto it = data_.find(name);
  if (it != data_.end()) return it->second.get();
  return nullptr;
}

Tensor Scope::GetTensor(const std::string& name) const {
  CheckVarNameValid(name);
  auto* var = FindVar(name);
  CHECK(var) << "No variable called [" << name << "] found";
  return absl::get<Tensor>(*var);
}

std::vector<absl::string_view> Scope::var_names() const {
  std::vector<absl::string_view> names;
  for (auto& item : data_) {
    names.push_back(item.first);
  }
  return names;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
