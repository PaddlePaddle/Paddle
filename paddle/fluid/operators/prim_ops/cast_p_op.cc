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
class CastPrimOp : public framework::OperatorBase {
 public:
  CastPrimOp(const std::string &type,
             const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator cast_p should not be excuted directly"));
  }
};

class CastPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of cast_p op.");
    AddOutput("Y", "(Tensor), The output tensor of cast_p op.");
    AddAttr<int>("dtype", "output data type");
    AddComment(R"DOC(Autograd primitive cast_p operator.)DOC");
  }
};

class CastPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    framework::VarDesc *x_var = PADDLE_GET(framework::VarDesc *, x_var_ptr);
    PADDLE_GET(framework::VarDesc *, y_var_ptr)->SetShape(x_var->GetShape());
  }
};

class CastPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto out_type = static_cast<framework::proto::VarType::Type>(
        PADDLE_GET_CONST(int, ctx->GetAttr("dtype")));
    ctx->SetOutputDataType("Y", out_type);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(cast_p,
                  paddle::operators::CastPrimOp,
                  paddle::operators::CastPrimOpMaker,
                  paddle::operators::CastPrimOpShapeInference,
                  paddle::operators::CastPrimOpVarTypeInference);
