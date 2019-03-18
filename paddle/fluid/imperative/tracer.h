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

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/engine.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace imperative {

void CreateGradOp(const framework::OpDesc& op_desc,
                  const std::unordered_set<std::string>& no_grad_set,
                  const std::vector<framework::BlockDesc*>& grad_sub_block,
                  framework::OpDesc** grad_op_desc,
                  std::unordered_map<std::string, std::string>* grad_to_var);

void InitVar(const VarBase* var, framework::Variable* grad_var,
             platform::DeviceContext* dev_ctx);

platform::Place GetExpectedPlace(platform::Place place, VarBasePtrMap inputs);

class Tracer {
 public:
  explicit Tracer(framework::BlockDesc* root_block);

  virtual ~Tracer() {}

  std::set<std::string> Trace(OpBase* op, const VarBasePtrMap& inputs,
                              VarBasePtrMap* outputs,  // NOLINT
                              framework::AttributeMap attrs_map,
                              const platform::Place expected_place,
                              const bool stop_gradient = false);

  std::vector<VarBase*> PyTrace(OpBase* op, const std::vector<VarBase*>& inputs,
                                bool stop_gradient = false);

 private:
  platform::Place GetPlace(const VarBasePtrMap& inputs);

  framework::BlockDesc* root_block_;
};

}  // namespace imperative
}  // namespace paddle
