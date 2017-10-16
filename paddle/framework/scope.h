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
#include <string>
#include <unordered_map>

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/variable.h"
#include "paddle/platform/macros.h"
#include "paddle/platform/place.h"

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

  /// Create a sub-scope. Returns a reference other than a pointer so
  /// to prevent from manual deletion.
  /// Mark it to const because that new kid scope cannot change parent scope.
  Scope& NewScope() const;

  /// Create a variable with given name if it doesn't exist.
  Variable* Var(const std::string& name);

  /// Create a variable with a scope-unique name.
  Variable* Var();

  /// Find a variable in the scope or any of its ancestors.  Returns
  /// nullptr if cannot find.
  Variable* FindVar(const std::string& name) const;

  const Scope& parent() const { return *parent_; }

  /// Find the scope or an ancestor scope that contains the given variable.
  const Scope* FindScope(const Variable* var) const;

  /// Drop all kids scopes belonged to this scope.
  void DropKids();

 private:
  // Call Scope::NewScope for a sub-scope.
  explicit Scope(Scope const* parent) : parent_(parent) {}

  std::unordered_map<std::string, Variable*> vars_;
  mutable std::list<Scope*> kids_;
  Scope const* parent_{nullptr};

  DISABLE_COPY_AND_ASSIGN(Scope);
};

framework::Scope& GetGlobalScope();

// template <typename T>
// void SetFeedVariable(const std::vector<T>& input, const Lod& lod,
//   const std::vector<int64_t>& dims,
//   const std::string& var_name, size_t index) {
//   Variable* g_feed_value = GetGlobalScope().Var("var_name");
//   // feed variable holds vector<LodTensor>
//   auto& feed_inputs =
//       *(g_feed_value->GetMutable<
// std::vector<paddle::framework::LoDTensor>>());
//   if (index >= feed_inputs.size()) {
//     feed_inputs.resize(index);
//   }
//   // copy tensor
//   T* dst = feed_inputs[index].mutable_data<T>(make_ddim(dims),
//     platform::CPUPlace());
//   memcpy(dst, inputs[i].data(), inputs[i].size() * sizeof(T));
//   // copy lod
//   feed_inputs[index].set_lod(lod);
// }

template <typename T>
void SetFeedVariable(const LoDTensor& input, const std::string& var_name,
                     size_t index) {
  std::cout << "into SetFeedVariable" << std::endl;
  std::cout << var_name << std::endl;
  std::cout << index << std::endl;
  Variable* g_feed_value = GetGlobalScope().Var(var_name);
  auto& feed_inputs =
      *(g_feed_value->GetMutable<std::vector<paddle::framework::LoDTensor>>());
  if (index >= feed_inputs.size()) {
    feed_inputs.resize(index + 1);
  }
  // shared data with input tensor
  feed_inputs[index].ShareDataWith<T>(input);
  // set lod
  feed_inputs[index].set_lod(input.lod());
}

// template <typename T>
// std::vector<T> GetFetchVariable(const std::string& var_name, size_t index) {
//   Variable* g_fetch_value = GetGlobalScope().Var(var_name);
//   auto& fetch_outputs =
//       *(g_fetch_value->GetMutable<
// std::vector<paddle::framework::LoDTensor>>());
//   std::vector<T> result;
//   result.resize(fetch_outputs[index].numel());
//   memcpy(result.data(), fetch_outputs[i].data<T>(),
//            fetch_outputs[i].numel() * sizeof(T));
// }

template <typename T>
LoDTensor& GetFetchVariable(const std::string& var_name, size_t index) {
  Variable* g_fetch_value = GetGlobalScope().Var(var_name);
  auto& fetch_outputs =
      *(g_fetch_value->GetMutable<std::vector<paddle::framework::LoDTensor>>());
  std::cout << "into GetFetchVariable" << std::endl;
  PADDLE_ENFORCE_LT(index, fetch_outputs.size());
  return fetch_outputs[index];
}

}  // namespace framework
}  // namespace paddle
