/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class Pad3dOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Pad3d");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Pad3d");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dim.size(), 5,
                      platform::errors::InvalidArgument(
                          "The size of Input(X)'s dimension should be equal to "
                          "5, but received %d. ",
                          x_dim.size()));

    std::vector<int64_t> out_dims(x_dim.size());
    auto data_format = ctx->Attrs().Get<std::string>("data_format");
    out_dims[0] = x_dim[0];
    if (ctx->HasInput("Paddings")) {
      auto paddings_dim = ctx->GetInputDim("Paddings");
      PADDLE_ENFORCE_EQ(paddings_dim.size(), 1,
                        platform::errors::InvalidArgument(
                            "Size of Input(Paddings)'s dimension should be "
                            "equal to 1, but received %d.",
                            paddings_dim.size()));
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(paddings_dim[0], 6,
                          platform::errors::InvalidArgument(
                              "Shape of Input(Paddings) should be equal to "
                              "[6], but received [%d].",
                              paddings_dim[0]));
      }
      out_dims[1] = x_dim[1];
      out_dims[2] = x_dim[2];
      out_dims[3] = x_dim[3];
    } else {
      auto paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
      PADDLE_ENFORCE_EQ(
          paddings.size(), 6,
          platform::errors::InvalidArgument(
              "Size of paddings should be equal to 4, but received %d.",
              static_cast<int>(paddings.size())));
      if (data_format == "NCDHW") {
        out_dims[1] = x_dim[1];  // channel
        out_dims[2] = ((!ctx->IsRuntime()) && (x_dim[2] < 0))
                          ? x_dim[2]
                          : (x_dim[2] + paddings[4] + paddings[5]);  // depth

        out_dims[3] = ((!ctx->IsRuntime()) && (x_dim[3] < 0))
                          ? x_dim[3]
                          : (x_dim[3] + paddings[2] + paddings[3]);  // height

        out_dims[4] = ((!ctx->IsRuntime()) && (x_dim[4] < 0))
                          ? x_dim[4]
                          : (x_dim[4] + paddings[0] + paddings[1]);  // width
      } else {                                                       // NDHWC
        out_dims[4] = x_dim[4];                                      // channel

        out_dims[1] = ((!ctx->IsRuntime()) && (x_dim[1] < 0))
                          ? x_dim[1]
                          : (x_dim[1] + paddings[4] + paddings[5]);  // depth
        out_dims[2] = ((!ctx->IsRuntime()) && (x_dim[2] < 0))
                          ? x_dim[2]
                          : (x_dim[2] + paddings[2] + paddings[3]);  // height
        out_dims[3] = ((!ctx->IsRuntime()) && (x_dim[3] < 0))
                          ? x_dim[3]
                          : (x_dim[3] + paddings[0] + paddings[1]);  // width
      }
    }

    ctx->SetOutputDim("Out", phi::make_ddim(out_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class Pad3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad3d op. "
             "The input should be a 5-D tensor with formate NCDHW or NDHWC.");
    AddOutput("Out",
              "The output of pad3d op. "
              "A tensor with the same shape as X.");
    AddInput("Paddings",
             "A 1-D tensor to describe the padding rules."
             "paddings=[0, 1, 2, 3, 4, 5] means "
             "padding 0 column to left, 1 column to right, "
             "2 row to top, 3 row to bottom, 4 depth to front "
             "and 5 depth to back. Size of paddings must be 6.")
        .AsDispensable();
    AddAttr<std::vector<int>>(
        "paddings",
        "(vector<int>) "
        "A list<int> to describe the padding rules."
        "paddings=[0, 1, 2, 3, 4, 5] means "
        "padding 0 column to left, 1 column to right, "
        "2 row to top, 3 row to bottom, 4 depth to front "
        "and 5 depth to back. Size of paddings must be 6.");
    AddAttr<float>("value",
                   "(float, default 0.0) "
                   "The value to fill the padded areas in constant mode.")
        .SetDefault(0.0f);
    AddAttr<std::string>(
        "mode",
        "(string, default constant) "
        "Four modes: constant(default), reflect, replicate, circular.")
        .SetDefault("constant");
    AddAttr<std::string>(
        "data_format",
        "(string, default NCDHW) Only used in "
        "An optional string from: \"NDHWC\", \"NCDHW\". "
        "Defaults to \"NDHWC\". Specify the data format of the input data.")
        .SetDefault("NCDHW");
    AddComment(R"DOC(
Pad3d Operator.
Pad 3-d images according to 'paddings' and 'mode'. 
If mode is 'reflect', paddings[0] and paddings[1] must be no greater
than width-1. The height and depth dimension have the same condition.

Given that X is a channel of image from input:

X = [[[[[1, 2, 3],
     [4, 5, 6]]]]]

Case 0:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'constant'
pad_value = 0

Out = [[[[[0. 0. 0. 0. 0. 0. 0.]
          [0. 0. 1. 2. 3. 0. 0.]
          [0. 0. 4. 5. 6. 0. 0.]
          [0. 0. 0. 0. 0. 0. 0.]]]]]

Case 1:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'reflect'

Out = [[[[[6. 5. 4. 5. 6. 5. 4.]
          [3. 2. 1. 2. 3. 2. 1.]
          [6. 5. 4. 5. 6. 5. 4.]
          [3. 2. 1. 2. 3. 2. 1.]]]]]

Case 2:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'replicate'

Out = [[[[[1. 1. 1. 2. 3. 3. 3.]
          [1. 1. 1. 2. 3. 3. 3.]
          [4. 4. 4. 5. 6. 6. 6.]
          [4. 4. 4. 5. 6. 6. 6.]]]]]

Case 3:

paddings = [2, 2, 1, 1, 0, 0],
mode = 'circular'

Out = [[[[[5. 6. 4. 5. 6. 4. 5.]
          [2. 3. 1. 2. 3. 1. 2.]
          [5. 6. 4. 5. 6. 4. 5.]
          [2. 3. 1. 2. 3. 1. 2.]]]]]

)DOC");
  }
};

class Pad3dOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Pad3d@Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "Pad3d@Grad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class Pad3dOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    bind->SetInput("X", this->Input("X"));
    if (this->HasInput("Paddings")) {
      bind->SetInput("Paddings", this->Input("Paddings"));
    }
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("pad3d_grad");
  }
};

template <typename T>
class Pad3dOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    if (this->HasInput("Paddings")) {
      grad_op->SetInput("Paddings", this->Input("Paddings"));
    }
    grad_op->SetType("pad3d");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(Pad3dOpGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(pad3d, Pad3dInferShapeFunctor,
                            PD_INFER_META(phi::Pad3dInferMeta));

REGISTER_OPERATOR(pad3d, ops::Pad3dOp, ops::Pad3dOpMaker,
                  ops::Pad3dOpGradMaker<paddle::framework::OpDesc>,
                  ops::Pad3dOpGradMaker<paddle::imperative::OpBase>,
                  Pad3dInferShapeFunctor);
REGISTER_OPERATOR(pad3d_grad, ops::Pad3dOpGrad,
                  ops::Pad3dOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::Pad3dOpDoubleGradMaker<paddle::imperative::OpBase>,
                  ops::Pad3dOpGradNoNeedBufferVarsInferer);
