// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
class InferShapeContext;
// class OpDesc;
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class ReshapePrimOp : public framework::OperatorBase {
 public:
  ReshapePrimOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    return;
  }
}

class ReshapePrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of reshape_p op.");
    AddOutput("Y", "(Tensor), The output tensor of reshape_p op.");
    AddAttr<std::vector<size_t>>("shape", "The target shape to reshape to.");
    AddComment(R"DOC(TODO
)DOC");
  }
};

class ReshapePrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    ctx->HasInputs("X");
    ctx->HasOutputs("Y");
    std::vector<framework::InferShapeVarPtr> in_var_ptrs =
        ctx->GetInputVarPtrs("X");
    std::vector<framework::InferShapeVarPtr> out_var_ptrs =
        ctx->GetOutputVarPtrs("Y");

    framework::VarDesc *in_var =
        BOOST_GET(framework::VarDesc *, in_var_ptrs[0]);

    // TODO(lml): add correct infershape
    BOOST_GET(framework::VarDesc *, out_var_ptrs[0])
        ->SetShape(in_var->GetShape());
  }
};

class ReshapePrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto in_names = Input(ctx, "X");
    auto out_names = Output(ctx, "Y");
    SetType(ctx, out_names[0], GetType(ctx, in_names[0]));
    SetDataType(ctx, out_names[0], GetDataType(ctx, in_names[0]));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(reshape_p, paddle::operators::ReshapePrimOp,
                  paddle::operators::ReshapePrimOpShapeInference,
                  paddle::operators::ReshapePrimOpVarTypeInference);
