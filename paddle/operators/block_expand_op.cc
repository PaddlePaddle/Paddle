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
    using namespace framework;
    PADDLE_ENFORCE(ctx->HasInput("input"),
                   "Input of BlockExpandOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of BlockExpandOp op should not be null.");

    auto in_dim = ctx->GetInputDim("input");
    PADDLE_ENFORCE_EQ(in_dim.size(), 4, "Input format  must be NCHW.");
    PADDLE_ENFORCE_GE(in_dim[0], 1, "Input batchsize must >= 1.");

    ctx->ShareLoD("X", /*->*/ "Out");

    // ctx->SetOutputDim("Out", {1});
  }
};

class BlockExpandOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BlockExpandOpMaker(framework::OpProto* proto,
                     framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "The input of block_expand op");
    AddOutput("out", "The output of block_expand op");
    AddAttr<int>("block_height",
                 R"DOC(
        )DOC");
    AddAttr<int>("block_width",
                 R"DOC(
        )DOC");
    AddAttr<int>("stride_height",
                 R"DOC(
        )DOC");
    AddAttr<int>("stride_width",
                 R"DOC(
        )DOC");
    AddAttr<int>("padding_height",
                 R"DOC(
        )DOC");
    AddAttr<int>("padding_width",
                 R"DOC(
        )DOC");
    AddComment(R"DOC(
Expand feature map to minibatch matrix.
- matrix width is: blockH_ * blockW_ * channels_
- matirx height is: outputH_ * outputW_

outputH\_ = 1 + (2paddingH\_ + imgSizeH\_ - blockH\_ + strideH\_ - 1) /
            strideH\_ \\
outputW\_ = 1 + (2paddingW\_ + imgSizeW\_ - blockW\_ + strideW\_ - 1) /
            strideW\_

The expand method is the same with ExpandConvLayer, but saved the transposed
value. After expanding, output_.sequenceStartPositions will store timeline.
The number of time steps are outputH_outputW_ and the dimension of each
time step is blockH_ * blockW_ * channels_. This layer can be used after
convolution neural network, and before recurrent neural network.
)DOC");
  }
};

class BlockExpandGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(block_expand, ops::BlockExpandOp, ops::BlockExpandOpMaker,
            block_expand_grad, ops::BlockExpandOpGrad);
REGISTER_OP_CPU_KERNEL(
    block_expand, ops::BlockExpanddKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    block_expand_grad,
    ops::BlockExpandGradKernel<paddle::platform::CPUPlace, float>);
