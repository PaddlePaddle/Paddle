/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/im2sequence_op.h"
#include <vector>

namespace paddle {
namespace operators {

class Im2SequenceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Im2SequenceOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of Im2SequenceOp op should not be null.");

    auto in_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(in_dim.size(), 4,
                      "Input(X) format must be 4D tensor, eg., NCHW.");

    auto kernels = ctx->Attrs().Get<std::vector<int>>("kernels");
    auto strides = ctx->Attrs().Get<std::vector<int>>("strides");
    auto paddings = ctx->Attrs().Get<std::vector<int>>("paddings");

    int batch_size = in_dim[0];
    int img_channels = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];

    int output_height = Im2SeqOutputSize(img_height, kernels[0], paddings[0],
                                         paddings[2], strides[0]);
    int output_width = Im2SeqOutputSize(img_width, kernels[1], paddings[1],
                                        paddings[3], strides[1]);

    ctx->SetOutputDim("Out", {batch_size * output_height * output_width,
                              img_channels * kernels[0] * kernels[1]});
  }
};

class Im2SequenceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input tensor has NCHW format."
             "N: batch size"
             "C: channels"
             "H: height"
             "W: width");
    AddOutput("Out", "(LodTensor) The output data of im2sequence op,");
    AddAttr<std::vector<int>>("kernels",
                              "(vector<int>), the "
                              "kernels(kernel_height, kernel_width)");
    AddAttr<std::vector<int>>("strides",
                              "(vector<int> default:{1, 1}), the "
                              "strides(h_stride, w_stride)")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector<int> default:{0, 0, 0, 0}), the "
                              "paddings(up_pad, left_pad, down_pad, right_pad)")
        .SetDefault({0, 0, 0, 0});
    AddComment(R"DOC(
This op uses kernels to scan images and converts these images to sequences.
After expanding, The number of time steps are output_height * output_width
and the dimension of each time step is kernel_height * kernel_width * channels,
in which:

output_height =
    1 + (padding_height + padding_down + img_height - kernel_height + stride_height - 1) /
            stride_height;
output_width =
    1 + (padding_left + padding+right + img_width - kernel_width + stride_width - 1) /
            stride_width;

This op can be used after convolution neural network, and before recurrent neural network.

Given:

x = [[[[ 6.  2.  1.]
       [ 8.  3.  5.]
       [ 0.  2.  6.]]

      [[ 2.  4.  4.]
       [ 6.  3.  0.]
       [ 6.  4.  7.]]]

     [[[ 6.  7.  1.]
       [ 5.  7.  9.]
       [ 2.  4.  8.]]

      [[ 1.  2.  1.]
       [ 1.  3.  5.]
       [ 9.  0.  8.]]]]
x.dims = {2, 2, 3, 3}

And:

kernels = [2, 2]
strides = [1, 1]
paddings = [0, 0, 0, 0]

Then:

output.data = [[ 6.  2.  8.  3.  2.  4.  6.  3.]
               [ 2.  1.  3.  5.  4.  4.  3.  0.]
               [ 8.  3.  0.  2.  6.  3.  6.  4.]
               [ 3.  5.  2.  6.  3.  0.  4.  7.]
               [ 6.  7.  5.  7.  1.  2.  1.  3.]
               [ 7.  1.  7.  9.  2.  1.  3.  5.]
               [ 5.  7.  2.  4.  1.  3.  9.  0.]
               [ 7.  9.  4.  8.  3.  5.  0.  8.]]
output.dims = {8, 9}
output.lod = [[0, 4, 8]]

)DOC");
  }
};

class Im2SequenceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(im2sequence, ops::Im2SequenceOp, ops::Im2SequenceOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(im2sequence_grad, ops::Im2SequenceGradOp);
REGISTER_OP_CPU_KERNEL(
    im2sequence,
    ops::Im2SequenceKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    im2sequence_grad,
    ops::Im2SequenceGradKernel<paddle::platform::CPUDeviceContext, float>);
