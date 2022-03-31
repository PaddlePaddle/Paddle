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
class MulPrimOp : public framework::OperatorBase {
 public:
  MulPrimOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator mul_p should not be excuted directly"));
  }
};

class MulPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of mul_p op.");
    AddInput("Y", "(Tensor), The input tensor of mul_p op.");
    AddOutput("Z", "(Tensor), The output tensor of mul_p op.");
  }
};

class MulPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr z_var_ptr = ctx->GetOutputVarPtrs("Z")[0];

    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);

    BOOST_GET(framework::VarDesc *, z_var_ptr)->SetShape(x_var->GetShape());
  }
};

class MulPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto z_name = Output(ctx, "Z")[0];
    SetType(ctx, z_name, GetType(ctx, x_name));
    SetDataType(ctx, z_name, GetDataType(ctx, x_name));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(mul_p, paddle::operators::MulPrimOp,
                  paddle::operators::MulPrimOpShapeInference,
                  paddle::operators::MulPrimOpVarTypeInference);
