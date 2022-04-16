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
class ReshapePrimOp : public framework::OperatorBase {
 public:
  ReshapePrimOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Prim operator reshape_p should not be excuted directly"));
  }
};

class ReshapePrimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of reshape_p op.");
    AddOutput("Y", "(Tensor), The output tensor of reshape_p op.");
    AddAttr<std::vector<int64_t>>(
        "shape", "(std::vector<int64_t>) Target shape of reshape_p operator.");
    AddComment(R"DOC(
Autograd primitive reshape_p operator.
)DOC");
  }
};

static int64_t product(const std::vector<int64_t> &shape) {
  int64_t rslt = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    rslt *= shape[i];
  }
  return rslt;
}

class ReshapePrimOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    framework::InferShapeVarPtr x_var_ptr = ctx->GetInputVarPtrs("X")[0];
    framework::InferShapeVarPtr y_var_ptr = ctx->GetOutputVarPtrs("Y")[0];
    framework::VarDesc *x_var = BOOST_GET(framework::VarDesc *, x_var_ptr);
    auto x_shape = x_var->GetShape();
    auto shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    PADDLE_ENFORCE_EQ(product(x_shape), product(shape),
                      platform::errors::InvalidArgument(
                          "The input tensor can't be reshaped to target shape, "
                          "the input tensor has %d elements but target shape "
                          "contains %d elements",
                          product(x_shape), product(shape)));
    BOOST_GET(framework::VarDesc *, y_var_ptr)->SetShape(shape);
  }
};

class ReshapePrimOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = Input(ctx, "X")[0];
    auto y_name = Output(ctx, "Y")[0];
    SetType(ctx, y_name, GetType(ctx, x_name));
    SetDataType(ctx, y_name, GetDataType(ctx, x_name));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(reshape_p, paddle::operators::ReshapePrimOp,
                  paddle::operators::ReshapePrimOpMaker,
                  paddle::operators::ReshapePrimOpShapeInference,
                  paddle::operators::ReshapePrimOpVarTypeInference);
