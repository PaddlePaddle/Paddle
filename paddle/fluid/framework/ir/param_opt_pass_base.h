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

#pragma once

#include <string>
#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * ParamOptPassBase is an basic interface for Passes that need to modify the
 * parameter before the runtime.
 * The Scope* need to be an attribute of graph called "param_scope", in
 * inference, this attribute will be set by inference/analysis automatically.
 *
 * A simple demo:
 *   class SomePass : public ParamOptPassBase {
 *    protected:
 *     void RegisterParamOperations() {
 *       ToRead("a_var_need_to_read_during_this_pass");
 *       ToWrite("a_var_need_to_write_during_this_pass");
 *       ToDrop("a_var_need_to_drop_after_this_pass_finishes");
 *       ToCreate("a_var_need_to_create_before_this_pass_starts");
 *     }
 *     void Operate() override() {
 *       // read a_var_need_to_read_during_this_pass
 *       // write a_var_need_to_write_during_this_pass
 *       // other operations.
 *     }
 *   };
 */
class ParamOptPassBase : public Pass {
 protected:
  // Use `Require`, `ToDrop`, `ToCreate` methods to register the parameters need
  // to operate on.
  virtual void RegisterParamOperations(Graph *graph, Scope *scope) const = 0;
  // Perform the opration.
  virtual void Operate(Graph *graph, Scope *scope) const = 0;

  // Register a parameter that will operate on (some check will be performed).
  void ToRead(const std::string &param) const;
  // Register a parameter that will write on.
  void ToWrite(const std::string &param) const;
  // Register a parameter to drop after this pass, this will done automatically.
  void ToDrop(const std::string &param) const;
  // Register a parameter to create before the pass run, this will done
  // automatically.
  void ToCreate(const std::string &param) const;
  //  Is called after a series of operation definitions.
  void CheckOrCreateParam(Graph *graph, Scope *scope) const;

  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;

  virtual ~ParamOptPassBase() = default;

 private:
  mutable std::array<std::unordered_set<std::string>, 4> reg_params_;
  const int kToRead = 0, kToWrite = 1, kToDrop = 2, kToCreate = 3;
  const std::array<std::string, 4> params_repr_{
      {"ToRead", "ToWrite", "ToDrop", "ToCreate"}};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
