// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {

class StaticPyLayerOp : public framework::OperatorBase {
 public:
  StaticPyLayerOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

  static const char kInputs[];
  static const char kOutputs[];
  static const char kScope[];
  static const char kSkipEagerDeletionVars[];
  static const char kBlocks[];

 protected:
  void CreateInterpreter(const platform::Place &dev_place,
                         const framework::BlockDesc &block,
                         framework::Scope *scope,
                         const std::vector<std::string> &skip_vars) const;

 protected:
  mutable std::shared_ptr<framework::InterpreterCore> core_{nullptr};
};

class StaticPyLayerForwardOpProtoMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(StaticPyLayerOp::kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(StaticPyLayerOp::kOutputs,
              "The output variables of the sub-block.")
        .AsDuplicable();
    // TODO(MarioLulab): Must Use std::vector here ?
    AddOutput(StaticPyLayerOp::kScope,
              "(std::vector<Scope*>) The scope of static pylayer block.");
    AddAttr<std::vector<framework::BlockDesc *>>(
        "blocks", "The blocks of PyLayer operator");
    AddComment(R"DOC(StaticPyLayer operator

TO-DO: added by luqi


)DOC");
  }
};

}  // namespace operators
}  // namespace paddle
