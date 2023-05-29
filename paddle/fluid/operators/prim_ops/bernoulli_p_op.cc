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
class VarDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class BernoulliPrimOp : public framework::OperatorBase {
 public:
  BernoulliPrimOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator bernoulli_p should not be excuted directly"));
  }
};

class BernoulliPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Y", "(Tensor), The output tensor of bernoulli_p op.");
    AddAttr<std::vector<int64_t>>(
        "shape", "(std::vector<int64_t>) The shape of output tensor.");
    AddAttr<int>("dtype", "(int) The dtype of output tensor.");
    AddAttr<float>("p", "(float) The probability of bernoulli distribution.");
    AddComment(R"DOC(
Autograd primitive bernoulli_p operator.
)DOC");
  }
};

class BernoulliPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    auto shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    PADDLE_GET(framework::VarDesc *, y_var_ptr)->SetShape(shape);
  }
};

class BernoulliPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto y_name = Output(ctx, "Y")[0];
    auto data_type = static_cast<framework::proto::VarType::Type>(
        PADDLE_GET_CONST(int, ctx->GetAttr("dtype")));
    SetDataType(ctx, y_name, data_type);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(bernoulli_p,
                  paddle::operators::BernoulliPrimOp,
                  paddle::operators::BernoulliPrimOpMaker,
                  paddle::operators::BernoulliPrimOpShapeInference,
                  paddle::operators::BernoulliPrimOpVarTypeInference);
