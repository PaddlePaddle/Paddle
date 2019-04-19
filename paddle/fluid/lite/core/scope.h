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

#pragma once
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/variable.h"

namespace paddle {
namespace lite {

class Scope final {
 public:
  Scope() {}
  ~Scope();

  Scope& NewScope() const;

  Variable* Var(const std::string& name);

  Variable* FindVar(const std::string& name) const;

  Variable* FindLocalVar(const std::string& name) const;

  const Scope* parent() const { return parent_; }

  // Following the legacy scope interface.
  std::vector<std::string> LocalVarNames() const {
    std::vector<std::string> keys;
    for (const auto& item : vars_) {
      keys.push_back(item.first);
    }
    return keys;
  }

 private:
  // Scope in `kids_` are owned by this class.
  mutable std::list<Scope*> kids_;
  const Scope* parent_{nullptr};
  std::unordered_map<std::string, std::unique_ptr<Variable>> vars_;
};

}  // namespace lite
}  // namespace paddle
