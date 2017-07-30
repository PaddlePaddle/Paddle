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

#include <list>
#include <map>
#include <string>

#include "paddle/framework/variable.h"

namespace paddle {
namespace framework {

class Scope;

/**
 * @brief Scope that manage all variables.
 *
 * Scope is an association of a name to Variable. All variables belong to
 * Scope. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`.
 * One net can run in different scopes and update different variable in the
 * scope.
 */
class Scope {
 public:
  Scope() {}
  ~Scope();

  // Create a sub-scope. Returns a reference other than a pointer so
  // to prevent from manual deletion.
  Scope& NewScope();

  // Create a variable with given name if it doesn't exist.
  Variable* NewVar(const std::string& name);

  // Create a variable with a scope-unique name.
  Variable* NewVar();

  // Find a variable in the scope or any of its ancestors.  Returns
  // nullptr if cannot find.
  Variable* FindVar(const std::string& name) const;

  // Find the scope or an ancestor scope that contains the given variable.
  Scope* FindScope(const Variable* var);

 private:
  // Call Scope::NewScope for a sub-scope.
  explicit Scope(Scope* parent) : parent_(parent) {}

  std::map<std::string, Variable*> vars_;
  std::list<Scope*> kids_;
  Scope* parent_{nullptr};
};

}  // namespace framework
}  // namespace paddle
