/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

#include <memory>

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
    auto rank_weight = dim_weight.size();
    PADDLE_ENFORCE(rank_weight >= 2 && rank_weight <= 5,
                   "The rank of Input(Weights) can only be 2, 3,"
                   "4, 5 for fc, conv1d, conv2d, conv3d layers.");

    int dim = ctx->Attrs().Get<int>("dim");
    int power_iters = ctx->Attrs().Get<int>("power_iters");
    PADDLE_ENFORCE(dim == 0 || dim == 1, "Attr(dim) can only be 0 or 1");
    PADDLE_ENFORCE(power_iters >= 0,
                   "Attr(power_iters) should be larger equal then 0");

    int h = dim_weight[dim];
    int w = 1;
    for (int i = 0; i < rank_weight; i++) {
      if (i != dim) {
        w *= dim_weight[i];
      }
    }
    auto dim_u = ctx->GetInputDim("U");
    auto dim_v = ctx->GetInputDim("V");
    PADDLE_ENFORCE_EQ(dim_u[0], h,
                      "Input(U) dims[0] should be equal to "
                      "Input(Weight) dims[Attr(dim)]");
    PADDLE_ENFORCE_EQ(
        dim_v[0], w,
        "Input(V) dims[0] should be equal to "
        "the product of Input(Weight) dims except dims[Attr(dim)]");

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
             "This can be a 2-D, 3-D, 4-D, 5-D tensor which is the "
             "weights of fc, conv1d, conv2d, conv3d layer.");
    AddInput("U",
             "The weight_u tensor of spectral_norm operator, "
             "This can be a 1-D tensor in shape [H, 1],"
             "H is the 1st dimentions of Weight after reshape"
             "corresponding by Attr(dim). As for Attr(dim) = 1"
             "in conv2d layer with weight shape [M, C, K1, K2]"
             "Weight will be reshape to [C, M*K1*K2], U will"
             "be in shape [C, 1].");
    AddInput("V",
             "The weight_v tensor of spectral_norm operator, "
             "This can be a 1-D tensor in shape [W, 1], "
             "W is the 2nd dimentions of Weight after reshape "
             "corresponding by Attr(dim). As for Attr(dim) = 1 "
             "in conv2d layer with weight shape [M, C, K1, K2] "
             "Weight will be reshape to [C, M*K1*K2], V will "
             "be in shape [M*K1*K2, 1].");
    AddOutput("Out",
              "The output weight tensor of spectral_norm operator, "
              "This tensor is in same shape with Input(Weight).");

    AddAttr<int>("dim",
                 "The index of dimension which should be permuted "
                 "to the first before reshaping Input(Weight) to "
                 "matrix, it should be set as 0 if Input(Weight) is "
                 "the weight of fc layer, and should be set as 1 if "
                 "Input(Weight) is the weight of conv layer, "
                 "default 0.")
        .SetDefault(0);
    AddAttr<int>("power_iters",
                 "number of power iterations to calculate "
                 "spectral norm, default 1.")
        .SetDefault(1);
    AddAttr<float>("eps",
                   "epsilon for numerical stability in "
                   "calculating norms")
        .SetDefault(1e-12);

    AddComment(R"DOC(
          This layer calculates the spectral normalization value of weight of
          fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
          tensor.

          Spectral normalization stabilizes the training of critic in GANs
          (Generative Adversarial Networks). This layer rescaling weight tensor
          with spectral normalize value.

          For spectral normalization calculations, we rescaling weight
          tensor with :math:`\sigma`, while :math:`\sigma{\mathbf{W}}` is

            $$\sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \\frac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}$$

          We calculate :math:`\sigma{\mathbf{W}}` through power iterations as

            $$
            \mathbf{v} = \mathbf{W}^{T} \mathbf{u}
            $$
            $$
            \mathbf{v} = \\frac{\mathbf{v}}{\|\mathbf{v}\|_2}
            $$
            $$
            \mathbf{u} = \mathbf{W}^{T} \mathbf{v}
            $$
            $$
            \mathbf{u} = \\frac{\mathbf{u}}{\|\mathbf{u}\|_2}
            $$

          And :math:`\sigma` should be

            $$\sigma{\mathbf{W}} = \mathbf{u}^{T} \mathbf{W} \mathbf{v}$$

          For details of spectral normalization, please refer to paper: 
          `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .
         )DOC");
  }
};

class SpectralNormGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("spectral_norm_grad");

    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetInput("Weight", Input("Weight"));
    op->SetInput("U", Input("U"));
    op->SetInput("V", Input("V"));

    op->SetOutput(framework::GradVarName("Weight"), InputGrad("Weight"));

    op->SetAttrMap(Attrs());

    return op;
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
                  ops::SpectralNormGradOpDescMaker);
REGISTER_OPERATOR(spectral_norm_grad, ops::SpectralNormOpGrad);
REGISTER_OP_CPU_KERNEL(
    spectral_norm,
    ops::SpectralNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SpectralNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    spectral_norm_grad,
    ops::SpectralNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SpectralNormGradKernel<paddle::platform::CPUDeviceContext, double>);
