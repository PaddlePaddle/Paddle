/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/im2sequence_op.h"

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
                      "Input(X) format  must be 4D tensor, eg., NCHW.");

    int block_height = ctx->Attrs().Get<int>("block_height");
    int block_width = ctx->Attrs().Get<int>("block_width");
    int stride_height = ctx->Attrs().Get<int>("stride_height");
    int stride_width = ctx->Attrs().Get<int>("stride_width");
    int padding_height = ctx->Attrs().Get<int>("padding_height");
    int padding_width = ctx->Attrs().Get<int>("padding_width");

    int batch_size = in_dim[0];
    int img_channels = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];

    int output_height = get_output_size(img_height, block_height, stride_height,
                                        padding_height);
    int output_width =
        get_output_size(img_width, block_width, stride_width, padding_width);

    ctx->SetOutputDim("Out", {batch_size * output_height * output_width,
                              img_channels * block_height * block_width});
    // TODO(wanghaoshuang): cal lod in complie time
  }
};

class Im2SequenceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Im2SequenceOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor)The input tensor has NCHW format."
             "N: batch size"
             "C: channels"
             "H: height"
             "W: width");
    AddOutput("Out", "(LodTensor)The output data of im2sequence op,");
    AddAttr<int>("block_height", "(int)height of block.");
    AddAttr<int>("block_width", "(int)width of block.");
    AddAttr<int>("stride_height", "(int)height of stride.");
    AddAttr<int>("stride_width", "(int)width of stride.");
    AddAttr<int>("padding_height", "(int)height of padding.");
    AddAttr<int>("padding_width", "(int)width of padding.");
    AddComment(R"DOC(
Convert feature map to minibatch matrix.
- matirx height is: output_height * output_width
- matrix width is: block_height * block_width * channels

output_height =
    1 + (2 * padding_height + img_height - block_height + stride_height - 1) /
            stride_height;
output_width =
    1 + (2 * padding_width + img_width - block_width + stride_width - 1) /
            stride_width;

After expanding, The number of time steps are output_height * output_width
and the dimension of each time step is block_height * block_width * channels.
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

block_height = 2
block_width = 2
stride_height = 1
stride_width = 1
padding_height = 0
padding_width = 0

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
REGISTER_OP(im2sequence, ops::Im2SequenceOp, ops::Im2SequenceOpMaker,
            im2sequence_grad, ops::Im2SequenceGradOp);
REGISTER_OP_CPU_KERNEL(
    im2sequence,
    ops::Im2SequenceKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    im2sequence_grad,
    ops::Im2SequenceGradKernel<paddle::platform::CPUDeviceContext, float>);
