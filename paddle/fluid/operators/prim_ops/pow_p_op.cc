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
namespace operators {
class PowPrimOp : public framework::OperatorBase {
 public:
  PowPrimOp(const std::string &type,
            const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator pow_p should not be excuted directly"));
  }
};

class PowPrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The base of pow_p op.");
    AddInput("Y", "(Tensor), The exponents of pow_p op.");
    AddOutput("Z", "(Tensor), The output tensor of pow_p op.");
    AddComment(R"DOC(
Autograd primitive pow_p operator.
)DOC");
  }
};

class PowPrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetInputVarPtrs("Y")[0];
    framework::InferShapeVarPtr z_var_ptr = ctx->GetOutputVarPtrs("Z")[0];

    framework::VarDesc *x_var = PADDLE_GET(framework::VarDesc *, x_var_ptr);
    framework::VarDesc *y_var = PADDLE_GET(framework::VarDesc *, y_var_ptr);
    auto x_shape = x_var->GetShape();
    auto y_shape = y_var->GetShape();
    size_t x_rank = x_shape.size();
    size_t y_rank = y_shape.size();

    PADDLE_ENFORCE_EQ(x_rank,
                      y_rank,
                      platform::errors::InvalidArgument(
                          "The dimensions of two input tensor should be same, "
                          "but get %d and %d",
                          x_rank,
                          y_rank));
    for (size_t i = 0; i < x_rank; ++i) {
      PADDLE_ENFORCE_EQ(
          x_shape[i],
          y_shape[i],
          platform::errors::InvalidArgument(
              "The shape of two input tensor at dimension %d should be same, "
              "but get %d and %d",
              i,
              x_shape[i],
              y_shape[i]));
    }

    PADDLE_GET(framework::VarDesc *, z_var_ptr)->SetShape(x_shape);
  }
};

class PowPrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto y_name = Input(ctx, "Y")[0];
    auto z_name = Output(ctx, "Z")[0];
    auto x_type = GetType(ctx, x_name);
    auto y_type = GetType(ctx, y_name);
    auto x_dtype = GetDataType(ctx, x_name);

    PADDLE_ENFORCE_EQ(x_type,
                      y_type,
                      platform::errors::InvalidArgument(
                          "The type of two input tensor should be same, "
                          "but get %d and %d",
                          x_type,
                          y_type));

    SetType(ctx, z_name, x_type);
    SetDataType(ctx, z_name, x_dtype);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(pow_p,
                  paddle::operators::PowPrimOp,
                  paddle::operators::PowPrimOpMaker,
                  paddle::operators::PowPrimOpShapeInference,
                  paddle::operators::PowPrimOpVarTypeInference);
