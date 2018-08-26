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

#include "param_opt_pass_base.h"
namespace paddle {
namespace framework {
namespace ir {

void ParamOptPassBase::ToRead(const std::string &param) const {
  PADDLE_ENFORCE(
      !reg_params_[kToWrite].count(param),
      "one parameter can not be registered to be both read and written");
  reg_params_[kToRead].insert(param);
}
void ParamOptPassBase::ToWrite(const std::string &param) const {
  PADDLE_ENFORCE(
      !reg_params_[kToRead].count(param),
      "one parameter can not be registered to be both read and written");
  reg_params_[kToWrite].insert(param);
}
void ParamOptPassBase::ToCreate(const std::string &param) const {
  PADDLE_ENFORCE(!reg_params_[kToDrop].count(param),
                 "one parameter can't be registered for both create and drop");
  reg_params_[kToCreate].insert(param);
}
void ir::ParamOptPassBase::ToDrop(const std::string &param) const {
  PADDLE_ENFORCE(!reg_params_[kToCreate].count(param),
                 "one parameter can't be registered for both create and drop");
  reg_params_[kToDrop].insert(param);
}

void ParamOptPassBase::CheckOrCreateParam(Graph *graph, Scope *scope) const {
  bool any_one_not_empty = false;
  for (size_t i = 0; i < reg_params_.size(); i++) {
    if (!reg_params_[i].empty()) {
      any_one_not_empty = true;
      break;
    }
  }
  PADDLE_ENFORCE(any_one_not_empty);

  // Check all the parameters to operate on exist in the scope.
  for (int i = kToRead; i <= kToDrop; i++) {
    for (auto &param : reg_params_[i]) {
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(param),
          "The variable [%s] %s required %s doesn't exist in the scope", param,
          params_repr_[i]);
    }
  }
  // Create the new parameters.
  for (auto &param : reg_params_[kToCreate]) {
    PADDLE_ENFORCE(!scope->FindVar(param),
                   "Cannot create the parameter [%s], already exists.", param);
    VLOG(4) << "to create parameter " << param;
    scope->Var(param);
  }
}

std::unique_ptr<ir::Graph> ir::ParamOptPassBase::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  // Require any one parameter to modify.
  // Run modify.
  PADDLE_ENFORCE(
      graph->Has("param_scope"),
      "PassOptPass require the graph has the [param_scope] attribute");

  auto *scope = graph->Get<Scope *>("param_scope");
  RegisterParamOperations(graph.get(), scope);
  Operate(graph.get(), scope);

  // Delete all the parameters need to drop.
  for (auto &param : reg_params_[kToDrop]) {
    VLOG(4) << "erase param " << param;
    scope->EraseVars({param});
  }

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
