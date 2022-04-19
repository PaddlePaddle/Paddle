// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>

#include <memory>
#include <string>
#include <vector>

#include "paddle/infrt/common/macros.h"
#include "paddle/infrt/paddle/tensor.h"
#include "paddle/infrt/support/variant.h"

namespace infrt {
namespace paddle {

using _Variable = Variant<Tensor>;

struct _Tensor_;

class Scope {
 public:
  static std::shared_ptr<Scope> Create() { return std::make_shared<Scope>(); }

  //! Get or create a variable.
  template <typename T>
  _Variable* Var(const std::string& name);

  //! Find a variable, get null if not exists.
  _Variable* FindVar(const std::string& name) const;

  Tensor GetTensor(const std::string& name) const;

  //! Get variable names.
  std::vector<std::string> var_names() const;

  Scope() = default;

 private:
  std::unordered_map<std::string, std::unique_ptr<_Variable>> data_;

  INFRT_DISALLOW_COPY_AND_ASSIGN(Scope);
};

template <typename T>
_Variable* Scope::Var(const std::string& name) {
  VLOG(4) << "Scope insert Var [" << name << "]";
  _Variable* x = FindVar(name);
  if (x) return x;
  auto* data = new _Variable(T());
  data_[name].reset(data);
  return data;
}

}  // namespace paddle
}  // namespace infrt
