/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace operators {

class FoldOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Tensor, "
             "the input of fold op. "
             "The format of X is [N, C_in*filter_height*filter_width, L]"
             "where N is the batch size, C_in is the channeled vector, "
             "filter_height and filter_width is height and width of the filtering kernel and L is the blocks amount");
    AddOutput(
        "Y",
        "Tensor, "
        "the output of fold op. "
        "The format of Y is [N, C_in,output_size[0],output_size[1],… ], "
        "where N is the batch size, C_in is the channeled vector, "
        "output_size is the spatial shape of the large containing tensor of the sliding local blocks");

    AddAttr<std::vector<int>>(
        "output_sizes", "vector<int>, the shape of the spatial dimensions of the output.");

    AddAttr<std::vector<int>>(
        "kernel_sizes", "vector<int>, the kernel sizes of the convolution operator.");

    AddAttr<std::vector<int>>(
        "dilations", "vector<int>, the dilations of the convolution operator.");

    AddAttr<std::vector<int>>(
        "paddings", "vector<int>, the paddings applied to pad the feature map.");

    AddAttr<std::vector<int>>(
        "strides", "vector<int>, the strides of the convolution operator.");


    AddComment(R"DOC(
**Fold Operator**

This Operator is used to calculates each combined value in the resulting large tensor by summing
all values from all containing blocks., also known as col2im when operated on batched 2D image tensor.
col2im, which rearrange matrix columns into blocks, can be taken as reverse operation to im2col.
    )DOC");
  }
};

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
    std::vector<int> output_sizes = ctx->Attrs().Get<std::vector<int>>("output_sizes");
    std::vector<int> kernel_sizes = ctx->Attrs().Get<std::vector<int>>("kernel_sizes");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx->Attrs().Get<std::vector<int>>("dilations");

    // Only [N, C, L] input supported now-
    PADDLE_ENFORCE_EQ(
        in_dims.size(), 3,
        platform::errors::InvalidArgument(
            "Input should be 3-D tensor of format [N, C, L], but get %u",
            in_dims.size()));

    PADDLE_ENFORCE_EQ(
        output_sizes.size(), 2,
        platform::errors::InvalidArgument(
            "Output size should be 2-D tensor of format [X, Y], but get %u",
            output_sizes.size()));

    PADDLE_ENFORCE_EQ(
        kernel_sizes.size(), 2,
        platform::errors::InvalidArgument(
            "Kernel size should be 2-D tensor of format [X, Y], but get %u",
            kernel_sizes.size()));

    PADDLE_ENFORCE_EQ(
        strides.size(), 2,
        platform::errors::InvalidArgument(
            "Strides size should be 2-D tensor of format [X, Y], but get %u",
            strides.size()));

    PADDLE_ENFORCE_EQ(
        paddings.size(), 2 * strides.size(),
        platform::errors::InvalidArgument(
            "The dims of paddings should be 2 times of that of strides. "
            "But recieved dims(paddings: %u) != 2*dims(strides: %u).",
            paddings.size(), strides.size()));

    PADDLE_ENFORCE_EQ(
        dilations.size(), 2,
        platform::errors::InvalidArgument(
            "Dilations size should be 2-D tensor of format [X, Y], but get %u",
            dilations.size()));

    PADDLE_ENFORCE_EQ(
        strides.size(), dilations.size(),
        platform::errors::InvalidArgument(
            "The dims of strides should be the same with that of dilations. "
            "But recieved dims(strides: %u) != dims(dilations: %u).",
            strides.size(), dilations.size()));

    // check kernel_sizes
    PADDLE_ENFORCE_GT(kernel_sizes[0], 0,
                      platform::errors::InvalidArgument(
                          "The `kernel_sizes` should be greater than zero, "
                          "but recieved kernel_height: %d kernel_width: %d.",
                          kernel_sizes[0], kernel_sizes[1]));
    PADDLE_ENFORCE_GT(kernel_sizes[1], 0,
                      platform::errors::InvalidArgument(
                          "The `kernel_sizes` should be greater than zero, "
                          "but recieved kernel_height: %d kernel_width: %d.",
                          kernel_sizes[0], kernel_sizes[1]));
    // check strides
    PADDLE_ENFORCE_GT(strides[0], 0,
                      platform::errors::InvalidArgument(
                          "The `strides` should be greater than zero, "
                          "but recieved strides_height: %d strides_width: %d.",
                          strides[0], strides[1]));
    PADDLE_ENFORCE_GT(strides[1], 0,
                      platform::errors::InvalidArgument(
                          "The `strides` should be greater than zero, "
                          "but recieved strides_height: %d strides_width: %d.",
                          strides[0], strides[1]));
    // check dilations
    PADDLE_ENFORCE_GT(
        dilations[0], 0,
        platform::errors::InvalidArgument(
            "The `dilations` should be greater than zero, "
            "but recieved dilations_height: %d dilations_width: %d.",
            dilations[0], dilations[1]));
    PADDLE_ENFORCE_GT(
        dilations[1], 0,
        platform::errors::InvalidArgument(
            "The `dilations` should be greater than zero, "
            "but recieved dilations_height: %d dilations_width: %d.",
            dilations[0], dilations[1]));

    // check output_sizes
    PADDLE_ENFORCE_GT(output_sizes[0], 0,
                      platform::errors::InvalidArgument(
                          "The `output_sizes` should be greater than zero, "
                          "but recieved output_height: %d output_width: %d.",
                          output_sizes[0], output_sizes[1]));
    PADDLE_ENFORCE_GT(output_sizes[1], 0,
                      platform::errors::InvalidArgument(
                          "The `output_sizes` should be greater than zero, "
                          "but recieved output_height: %d output_width: %d.",
                          output_sizes[0], output_sizes[1]));

    std::vector<int> out_dims;
    out_dims.push_back(in_dims[0]);

    int output_channels = in_dims[1] / (kernel_sizes[0] * kernel_sizes[1]);
    out_dims.push_back(output_channels);
    out_dims.push_back(output_sizes[0]);
    out_dims.push_back(output_sizes[1]);

    // check output height and width
    PADDLE_ENFORCE_GT(
        output_channels, 0,
        platform::errors::InvalidArgument(
            "The sliding blocks calculated from input spatial size (%d, %d), "
            "kernel_sizes (%d, %d), strides (%d, %d), dilations (%d, %d), "
            "is (%d, %d), which should be a positive integer.",
            in_dims[1], in_dims[2], kernel_sizes[0], kernel_sizes[1],
            strides[0], strides[1], dilations[0], dilations[1], output_sizes[0],
            output_sizes[1]));

    PADDLE_ENFORCE_EQ(
        in_dims[1] % (kernel_sizes[0] * kernel_sizes[1]), 0,
        platform::errors::InvalidArgument(
            "Expected size of input's dimension 1 to be divisible by the "
            "product of kernel_size (%u * %u), but got input.size(1)= %u",
            kernel_sizes[0], kernel_sizes[1], in_dims[1]));

    //check input length indims[2]
    int input_length = in_dims[2];
    int n_blocks_height =
        CalcOutputSize(output_sizes[0], kernel_sizes[0], dilations[0],
                       paddings[0], paddings[2], strides[0]);
    int n_blocks_width =
        CalcOutputSize(output_sizes[1], kernel_sizes[1], dilations[1],
                       paddings[1], paddings[3], strides[1]);

    PADDLE_ENFORCE_EQ(
        n_blocks_height * n_blocks_width, input_length,
        platform::errors::InvalidArgument(
            "expected size of input's dimension 2 to match the calculated number of "
            "sliding blocks n_blocks_height %u * n_blocks_width %u, bug got input.size(2)=%u",
            n_blocks_height, n_blocks_width, input_length));

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
    fold_grad,
    ops::FoldGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FoldGradOpKernel<paddle::platform::CPUDeviceContext, double>);
