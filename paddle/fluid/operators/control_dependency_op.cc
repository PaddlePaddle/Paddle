/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/operators/send_recv_util.h"

namespace paddle {
namespace operators {

class ControlDependencyOp : public framework::OperatorBase {
 public:
  ControlDependencyOp(const std::string& type,
                      const framework::VariableNameMap& inputs,
                      const framework::VariableNameMap& outputs,
                      const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {}
};

class ControlDependencyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Any) Inputs").AsDuplicable();
    AddOutput("Out", "(Any) Outputs").AsDuplicable();
    AddComment(R"DOC(
Control Dependency operator.
)DOC");
  }
};

class ControlDependencyOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(contro_dependency, ops::ControlDependencyOp,
                  paddle::framework::EmptyGradOpMaker,
                  ops::ControlDependencyOpMaker,
                  ops::ControlDependencyOpShapeInference);
