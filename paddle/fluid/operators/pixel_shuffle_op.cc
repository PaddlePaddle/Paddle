/*Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/pixel_shuffle_op.h"

namespace paddle {
namespace operators {

class PixelShuffleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of PixelShuffleOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of PixelShuffleOp should not be null.");

    auto input_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(input_dims.size() == 4, "The layout of input is NCHW.");
    auto upscale_factor = ctx->Attrs().Get<int>("upscale_factor");

    PADDLE_ENFORCE(input_dims[1] % (upscale_factor * upscale_factor) == 0,
                   "Upscale_factor should devide the number of channel");

    auto output_dims = input_dims;
    output_dims[0] = input_dims[0];
    output_dims[1] = input_dims[1] / (upscale_factor * upscale_factor);
    output_dims[2] = input_dims[2] * upscale_factor;
    output_dims[3] = input_dims[3] * upscale_factor;
    ctx->SetOutputDim("Out", output_dims);
  }
};

class PixelShuffleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor, default Tensor<float>), "
        "the input feature data of PixelShuffleOp, the layout is [N C H W].");
    AddOutput(
        "Out",
        "(Tensor, default Tensor<float>), the output of "
        "PixelShuffleOp. The layout is [N,C/factor^2,H*factor,W*factor].");
    AddAttr<int>("upscale_factor",
                 "the factor to increase spatial resolution by.")
        .SetDefault(1)
        .AddCustomChecker([](const int& upscale_factor) {
          PADDLE_ENFORCE_GE(upscale_factor, 1,
                            "upscale_factor should be larger than 0.");
        });

    AddComment(R"DOC(
		Pixel Shuffle operator
		This operator rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    		to a tensor of shape :math:`(C, H \times r, W \times r)`.

		This is useful for implementing efficient sub-pixel convolution
    		with a stride of :math:`1/r`.

		Please refer to the paper:
		 `Real-Time Single Image and Video Super-Resolution Using an Efficient 
		 Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_
    		by Shi et. al (2016) for more details. 

        )DOC");
  }
};

class PixelShuffleGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("pixel_shuffle_grad");
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetAttrMap(Attrs());
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

class PixelShuffleGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@Grad) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@Grad) should not be null");

    auto do_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    PADDLE_ENFORCE(do_dims.size() == 4, "The layout of input is NCHW.");

    auto upscale_factor = ctx->Attrs().Get<int>("upscale_factor");

    auto dx_dims = do_dims;
    dx_dims[0] = do_dims[0];
    dx_dims[1] = do_dims[1] * (upscale_factor * upscale_factor);
    dx_dims[2] = do_dims[2] / upscale_factor;
    dx_dims[3] = do_dims[3] / upscale_factor;
    ctx->SetOutputDim(framework::GradVarName("X"), dx_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(pixel_shuffle, ops::PixelShuffleOp, ops::PixelShuffleOpMaker,
                  ops::PixelShuffleGradMaker);

REGISTER_OPERATOR(pixel_shuffle_grad, ops::PixelShuffleGradOp);

REGISTER_OP_CPU_KERNEL(
    pixel_shuffle,
    ops::PixelShuffleOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PixelShuffleOpKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    pixel_shuffle_grad,
    ops::PixelShuffleGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PixelShuffleGradOpKernel<paddle::platform::CPUDeviceContext, double>);
