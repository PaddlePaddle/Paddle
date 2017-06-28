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

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/framework/variable.h"
#include "paddle/platform/assert.h"

namespace paddle {
namespace framework {

/**
 * Scope is an association of a name to Variable. All variables belong to
 * Scope. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`.
 * One net can run in different scopes and update different variable in the
 * scope.
 */
class Scope {
 public:
  explicit Scope(const std::shared_ptr<Scope>& parent = nullptr)
      : parent_(parent) {}

  /// Create Variable in this Scope. Failed if Variable already been
  /// created.
  Variable* CreateVariable(const std::string& name) {
    PADDLE_ASSERT(!HasVariable(name));
    vars_[name] = std::unique_ptr<Variable>(new Variable());
    return GetVariable(name);
  }

  /// Get Variable from this Scope, this function will recursive find Variable
  /// from it's parent scope. Return nullptr if not found.
  Variable* GetVariable(const std::string& name) const {
    auto it = vars_.find(name);
    if (it != vars_.end()) {
      return it->second.get();
    } else if (parent_ != nullptr) {
      return parent_->GetVariable(name);
    } else {
      return nullptr;
    }
  }

  /// Get Variable from scope, if Variable is not exist, creat one and return.
  Variable* GetOrCreateVariable(const std::string& name) {
    auto var = GetVariable(name);
    if (var) {
      return var;
    } else {
      return CreateVariable(name);
    }
  }

  /// Find if there is a Variable in this scope and it's parent scope
  bool HasVariable(const std::string& name) const {
    return (vars_.find(name) != vars_.end() ||
            (parent_ && parent_->HasVariable(name)));
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<Variable>> vars_;
  std::shared_ptr<Scope> parent_{nullptr};
};

}  // namespace framework
}  // namespace paddle
