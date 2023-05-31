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

#pragma once
#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>
#include <absl/types/any.h>
#include <absl/types/variant.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

using Variable = absl::variant<Tensor>;

struct _Tensor_;

class Scope {
 public:
  static std::shared_ptr<Scope> Create() { return std::make_shared<Scope>(); }

  //! Get or create a variable.
  template <typename T>
  Variable* Var(const std::string& name);

  // Erase a variable, check exists firstly
  void EraseVar(const std::string& name);

  //! Find a variable, get null if not exists.
  Variable* FindVar(const std::string& name) const;

  Tensor GetTensor(const std::string& name) const;

  //! Get variable names.
  std::vector<absl::string_view> var_names() const;

  Scope() = default;

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<Variable>> data_;

  CINN_DISALLOW_COPY_AND_ASSIGN(Scope);
};

template <typename T>
Variable* Scope::Var(const std::string& name) {
  VLOG(4) << "Scope insert Var [" << name << "]";
  Variable* x = FindVar(name);
  if (x) return x;
  auto* data = new Variable(T());
  data_[name].reset(data);
  return data;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
