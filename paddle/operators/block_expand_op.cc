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

#include "paddle/operators/block_expand_op.h"

namespace paddle {
namespace operators {

class BlockExpandOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    printf("op infershape\n");
    using namespace framework;
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input of BlockExpandOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output of BlockExpandOp op should not be null.");

    auto in_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(in_dim.size(), 4, "Input format  must be NCHW.");
    PADDLE_ENFORCE_GE(in_dim[0], 1, "Input batchsize must >= 1.");

    printf("op infershape2\n");
    int block_height = ctx->Attrs().Get<int>("blockHeight");
    int block_width = ctx->Attrs().Get<int>("blockWidth");
    int stride_height = ctx->Attrs().Get<int>("strideHeight");
    int stride_width = ctx->Attrs().Get<int>("strideWidth");
    int padding_height = ctx->Attrs().Get<int>("paddingHeight");
    int padding_width = ctx->Attrs().Get<int>("paddingWidth");

    int N = in_dim[0];
    int C = in_dim[1];
    int img_height = in_dim[2];
    int img_width = in_dim[3];

    int output_height = 0;
    int output_width = 0;

    get_blockexpand_output_shape(img_height, img_width, block_height,
                                 block_width, stride_height, stride_width,
                                 padding_height, padding_width, output_height,
                                 output_width);

    // The result of im2col is [output_height, output_width,
    // inputChannels, filterHeight, filterWidth], and it is easy to
    // reshape into [seqLength, stepSize], where seqLength is equal
    // output_height * output_width, stepSize is equal
    // input_channels * blockHeight * blockWidth
    printf("N:%d, o_h:%d o_w:%d C:%d b_h:%d b_w:%d\n", N, output_height,
           output_width, C, block_height, block_width);
    ctx->SetOutputDim(
        "Out", {N, output_height, output_width, C, block_height, block_width});

    // ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class BlockExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BlockExpandOpMaker(framework::OpProto* proto,
                     framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", R"DOC(
(Tensor)The input tensor has NCHW format.
    N: batch size
    C: channels
    H: height
    W: width
)DOC");
    printf("opmakeer\n");
    AddOutput("Out", "(LodTensor)The output data of block_expand op,");
    AddAttr<int>("blockHeight", "(int)height of block.");
    AddAttr<int>("blockWidth", "(int)width of block.");
    AddAttr<int>("strideHeight", "(int)height of stride.");
    AddAttr<int>("strideWidth", "(int)width of stride.");
    AddAttr<int>("paddingHeight", "(int)height of padding.");
    AddAttr<int>("paddingWidth", "(int)width of padding.");
    AddComment(R"DOC(
Expand feature map to minibatch matrix.
- matirx height is: output_height * output_width
- matrix width is: blockHeight * blockWidth * channels

output_height = 
    1 + (2 * paddingHeight + img_height - blockHeight + strideHeight - 1) /
            strideHeight;
output_width = 
    1 + (2 * paddingWidth + img_width - blockWidth + strideWidth - 1) /
            strideWidth;

The expand method is the same with ExpandConvLayer, but saved the transposed
value. After expanding, The number of time steps are output_height * output_width
and the dimension of each time step is blockHeight * blockWidth * channels.
This layer can be used after convolution neural network, and before recurrent neural network.
)DOC");
  }
};

class BlockExpandGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    using namespace framework;
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output of BlockExpandOp op should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto in_dim = ctx->GetInputDim("X");

    ctx->SetOutputDim(GradVarName("Out"), in_dim);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(block_expand, ops::BlockExpandOp, ops::BlockExpandOpMaker,
            block_expand_grad, ops::BlockExpandGradOp);
REGISTER_OP_CPU_KERNEL(
    block_expand, ops::BlockExpandKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    block_expand_grad,
    ops::BlockExpandGradKernel<paddle::platform::CPUPlace, float>);
