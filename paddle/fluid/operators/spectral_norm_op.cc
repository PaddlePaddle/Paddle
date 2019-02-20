/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/spectral_norm_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SpectralNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(Weight) of SpectralNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("U"),
                   "Input(U) of SpectralNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("V"),
                   "Input(V) of SpectralNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SpectralNormOp should not be null.");

    auto dim_weight = ctx->GetInputDim("Weight");
    auto weight_dimsize = dim_weight.size();
    PADDLE_ENFORCE(weight_dimsize >= 2 && weight_dimsize <= 5,
                   "The size of dims of Input(Weights) can only be 2, 3,"
                   "4, 5 for fc, conv1d, conv2d, conv3d layers.");

    int dim = ctx->Attrs().Get<int>("dim");
    int power_iters = ctx->Attrs().Get<int>("power_iters");
    PADDLE_ENFORCE(dim >= 0 && dim < weight_dimsize - 1,
                   "Attr(dim) should be larger equal 0 and less then the"
                   "size of dims of Input(Weights) - 1,");
    PADDLE_ENFORCE(power_iters >= 0,
                   "Attr(power_iters) should be larger equal then 0");

    ctx->SetOutputDim("Out", dim_weight);
    ctx->ShareLoD("Weight", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Weight")->type(),
                                   ctx.GetPlace());
  }
};

class SpectralNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Weight",
             "The input weight tensor of spectral_norm operator, "
             "This can be a 2-D, 3-D, 4-D, 5-D tensor which is the"
             "weights of fc, conv1d, conv2d, conv3d layer.");
    AddInput("U",
             "The weight_u tensor of spectral_norm operator, "
             "This can be a 1-D tensor in shape [H, 1],"
             "H is the 1st dimentions of Weight after reshape"
             "corresponding by Attr(dim).");
    AddInput("V",
             "The weight_u tensor of spectral_norm operator, "
             "This can be a 1-D tensor in shape [W, 1],"
             "W is the 2nd dimentions of Weight after reshape"
             "corresponding by Attr(dim).");
    AddOutput("Out",
              "The output weight tensor of spectral_norm operator, "
              "This tensor is in same shape with Input(Weight).");

    AddAttr<int>("dim",
                 "dimension corresponding to number of outputs,"
                 "default 0 for fc layer, and 1 for conv1d, conv2d, conv3d"
                 "layers")
        .SetDefault(0);
    AddAttr<int>("power_iters",
                 "number of power iterations to calculate"
                 "spectral norm, default is 1.")
        .SetDefault(1);
    AddAttr<float>("eps",
                   "epsilob for numerical stability in"
                   "calculating norms")
        .SetDefault(1e-12);

    AddComment(R"DOC(
          This operator samples input X to given output shape by using specified

          

         )DOC");
  }
};

class SpectralNormOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Weight"), "Input(Weight) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("U"), "Input(U) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("V"), "Input(V) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("Weight");
    if (ctx->HasOutput(framework::GradVarName("Weight"))) {
      ctx->SetOutputDim(framework::GradVarName("Weight"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Weight")->type(),
                                   ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(spectral_norm, ops::SpectralNormOp, ops::SpectralNormOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(spectral_norm_grad, ops::SpectralNormOpGrad);
REGISTER_OP_CPU_KERNEL(
    spectral_norm,
    ops::SpectralNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SpectralNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    spectral_norm_grad,
    ops::SpectralNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SpectralNormGradKernel<paddle::platform::CPUDeviceContext, double>);
