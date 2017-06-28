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

namespace paddle {
namespace framework {

/**
 * Scope is an association of a name to Variable. All variables belong to
 * `Scope`. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`.
 * One net can run in different scopes and update different variable in the
 * scope.
 */
class Scope {
 public:
  Scope() {}

  explicit Scope(const std::shared_ptr<Scope>& scope) : parent_(scope) {}

  ~Scope() {}

  // Create Variable in this Scope. Return error if Variable already been
  // created.
  Variable* CreateVariable(const std::string& name);

  // Get Variable from this Scope, this function will recursive find Variable
  // from it's parent scope. Return nullptr if not found.
  Variable* GetVariable(const std::string& name) const;

  // Find and return Variables in the scope it self.
  Variable* GetVarLocally(const std::string& name) const;

  // Find if there is a Variable in this scope and it's parent scope
  bool HasVariable(const std::string& name);

 private:
  std::unordered_map<std::string, std::unique_ptr<Variable>> vars_;
  std::shared_ptr<Scope> parent_{nullptr};
};

}  // namespace framework
}  // namespace paddle
