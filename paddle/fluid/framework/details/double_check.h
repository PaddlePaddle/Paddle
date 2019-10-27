/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace details {

class DoubleCheckOperator {
 public:
  explicit DoubleCheckOperator(const OperatorBase& base_op,
                               const OpHandleBase& base_handle)
      : base_op_(base_op), base_handle_(base_handle) {}
  void Run(const Scope& scope, const platform::Place& place);

 protected:
  void Diff(const Scope& scope, const platform::Place& place,
            const std::string& a, const std::string& b);

  void PrepareNameMap(Scope* scope, const platform::Place& place,
                      const framework::VariableNameMap& name_map,
                      framework::VariableNameMap* dst_name_map,
                      std::map<std::string, std::string>* diff_var_names,
                      const std::vector<VarHandleBase*>& var_handles);

  void Wait(const platform::Place& place);

  void GetCastInputAndOutputs(
      const Scope& scope, const platform::Place& place,
      const OperatorBase& base_op,
      std::map<std::string, std::string>* diff_var_names);

 private:
  const OperatorBase& base_op_;
  const OpHandleBase& base_handle_;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
