/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/fold_op.h"
#include "paddle/fluid/operators/unfold_op.h"

namespace paddle {
namespace operators {

class FoldOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of FoldOp should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Y"), true,
        platform::errors::NotFound("Output(Y) of FoldOp should not be null"));
    auto in_dims = ctx->GetInputDim("X");
    std::vector<int> output_sizes =
        ctx->Attrs().Get<std::vector<int>>("output_sizes");
    std::vector<int> kernel_sizes =
        ctx->Attrs().Get<std::vector<int>>("kernel_sizes");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> dilations =
        ctx->Attrs().Get<std::vector<int>>("dilations");

    PADDLE_ENFORCE_EQ(
        output_sizes.size(), 2,
        platform::errors::InvalidArgument(
            "It is expected output_size equals to 2, but got size %d",
            output_sizes.size()));
    PADDLE_ENFORCE_EQ(
        kernel_sizes.size(), 2,
        platform::errors::InvalidArgument(
            "It is expected kernel_size equals to 2, but got size %d",
            kernel_sizes.size()));
    PADDLE_ENFORCE_EQ(
        strides.size(), 2,
        platform::errors::InvalidArgument(
            "It is expected strides_size equals to 2, but got size %d",
            strides.size()));
    PADDLE_ENFORCE_EQ(
        paddings.size(), 4,
        platform::errors::InvalidArgument(
            "It is expected paddings_size equals to 4, but got size %d",
            paddings.size()));
    PADDLE_ENFORCE_EQ(
        dilations.size(), 2,
        platform::errors::InvalidArgument(
            "It is expected dilations_size equals to 2, but got size %d",
            dilations.size()));

    int output_height = output_sizes[0];
    int output_width = output_sizes[1];
    int kernel_height = kernel_sizes[0];
    int kernel_width = kernel_sizes[1];
    int dilation_height = dilations[0];
    int dilation_width = dilations[1];
    int stride_height = strides[0];
    int stride_width = strides[1];

    // check kernel_sizes
    PADDLE_ENFORCE_GT(kernel_height, 0,
                      platform::errors::InvalidArgument(
                          "The `kernel_sizes` should be greater than zero, "
                          "but recieved kernel_height: %d kernel_width: %d.",
                          kernel_sizes[0], kernel_sizes[1]));
    PADDLE_ENFORCE_GT(kernel_width, 0,
                      platform::errors::InvalidArgument(
                          "The `kernel_sizes` should be greater than zero, "
                          "but recieved kernel_height: %d kernel_width: %d.",
                          kernel_sizes[0], kernel_sizes[1]));
    // check strides
    PADDLE_ENFORCE_GT(stride_height, 0,
                      platform::errors::InvalidArgument(
                          "The `strides` should be greater than zero, "
                          "but recieved strides_height: %d strides_width: %d.",
                          strides[0], strides[1]));
    PADDLE_ENFORCE_GT(stride_width, 0,
                      platform::errors::InvalidArgument(
                          "The `strides` should be greater than zero, "
                          "but recieved strides_height: %d strides_width: %d.",
                          strides[0], strides[1]));
    // check dilations
    PADDLE_ENFORCE_GT(
        dilation_height, 0,
        platform::errors::InvalidArgument(
            "The `dilations` should be greater than zero, "
            "but recieved dilations_height: %d dilations_width: %d.",
            dilations[0], dilations[1]));
    PADDLE_ENFORCE_GT(
        dilation_width, 0,
        platform::errors::InvalidArgument(
            "The `dilations` should be greater than zero, "
            "but recieved dilations_height: %d dilations_width: %d.",
            dilations[0], dilations[1]));

    std::vector<int> out_dims;
    // batch_size
    out_dims.push_back(in_dims[0]);
    // output_plane
    int output_channels = in_dims[1] / (kernel_width * kernel_height);
    out_dims.push_back(output_channels);

    int blocks_height = (output_sizes[0] + 2 * paddings[0] -
                         (dilations[0] * (kernel_sizes[0] - 1) + 1)) /
                            strides[0] +
                        1;
    int blocks_width = (output_sizes[1] + 2 * paddings[1] -
                        (dilations[1] * (kernel_sizes[1] - 1) + 1)) /
                           strides[1] +
                       1;

    // check output height and width
    PADDLE_ENFORCE_GT(
        blocks_height, 0,
        platform::errors::InvalidArgument(
            "The sliding blocks calculated from input spatial size (%d, %d), "
            "kernel_sizes (%d, %d), strides (%d, %d), dilations (%d, %d), "
            "is (%d, %d), which should be a positive integer.",
            in_dims[2], in_dims[3], kernel_sizes[0], kernel_sizes[1],
            strides[0], strides[1], dilations[0], dilations[1], output_height,
            output_width));

    PADDLE_ENFORCE_GT(
        blocks_width, 0,
        platform::errors::InvalidArgument(
            "The sliding blocks calculated from input spatial size (%d, %d), "
            "kernel_sizes (%d, %d), strides (%d, %d), dilations (%d, %d), "
            "is (%d, %d), which should be a positive integer.",
            in_dims[2], in_dims[3], kernel_sizes[0], kernel_sizes[1],
            strides[0], strides[1], dilations[0], dilations[1], output_height,
            output_width));

    PADDLE_ENFORCE_EQ(
        blocks_height * blocks_width, in_dims[1],
        platform::errors::InvalidArgument(
            "Given input output_size (%d, %d), "
            "kernel_sizes (%d, %d), strides (%d, %d), dilations (%d, %d), "
            "which should be expected size of input's dimension "
            "2 to match the calculated number of %d * %d = %d, but got %d",
            output_height, output_width, kernel_sizes[0], kernel_sizes[1],
            strides[0], strides[1], dilations[0], dilations[1], blocks_height,
            blocks_width, blocks_height * blocks_width, in_dims[2]));

    out_dims.push_back(output_height);
    out_dims.push_back(output_width);
    ctx->SetOutputDim("Y", framework::make_ddim(out_dims));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class FoldOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, "
             "the input of fold op. "
             "The format of X is [N, C_in, L], "
             "where N is the batch size, C_in is the input channels, "
             "L is the length");
    AddOutput("Y",
              "Tensor, "
              "the output of unfold op. "
              "The format of Y is [N, C_out, output_height, output_width], "
              "where N is the batch size, "
              "C_in is the output channels of Y, output_height and "
              "output_width "
              "is the calculated height and width of output feature map.");
    AddAttr<std::vector<int>>(
        "output_sizes",
        "vector<int>, the output sizes of the convolution operator.");
    AddAttr<std::vector<int>>(
        "kernel_sizes",
        "vector<int>, the kernel sizes of the convolution operator.");
    AddAttr<std::vector<int>>(
        "strides", "vector<int>, the strides of the convolution operator.");
    AddAttr<std::vector<int>>(
        "paddings",
        "vector<int>, the paddings applied to pad the feature map.");
    AddAttr<std::vector<int>>(
        "dilations", "vector<int>, the dilations of the convolution operator.");
    AddComment(R"DOC(
**Fold Operator**

This Operator is used to combines an array of sliding local blocks into a large containing
tensor. also known as col2im when operated on batched 2D image tensor. Fold calculates each 
combined value in the resulting large tensor by summing all values from all containing blocks. 
Unfold extracts the values in the local blocks by copying from the large tensor. So, if the 
blocks overlap, they are not inverses of each other.
    )DOC");
  }
};

class FoldGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Y")), true,
        platform::errors::NotFound("The gradient of Y should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("The input X should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::NotFound("The gradient of X should not be null"));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }
};

template <typename T>
class FoldGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fold_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetInput("X", this->Input("X"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(FoldGradOpNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fold, ops::FoldOp, ops::FoldOpMaker,
                  ops::FoldGradMaker<paddle::framework::OpDesc>,
                  ops::FoldGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fold_grad, ops::FoldGradOp,
                  ops::FoldGradOpNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(
    fold, ops::FoldOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FoldOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    fold_grad, ops::FoldGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FoldGradOpKernel<paddle::platform::CPUDeviceContext, double>);
