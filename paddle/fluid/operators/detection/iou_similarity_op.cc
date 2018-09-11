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

#include "paddle/fluid/operators/detection/iou_similarity_op.h"

namespace paddle {
namespace operators {

class IOUSimilarityOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of IOUSimilarityOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of IOUSimilarityOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(x_dims.size(), 2UL, "The rank of Input(X) must be 2.");
    PADDLE_ENFORCE_EQ(x_dims[1], 4UL, "The shape of X is [N, 4]");
    PADDLE_ENFORCE_EQ(y_dims.size(), 2UL, "The rank of Input(Y) must be 2.");
    PADDLE_ENFORCE_EQ(y_dims[1], 4UL, "The shape of Y is [M, 4]");

    ctx->ShareLoD("X", /*->*/ "Out");
    ctx->SetOutputDim("Out", framework::make_ddim({x_dims[0], y_dims[0]}));
  }
};

class IOUSimilarityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, default LoDTensor<float>) "
             "Box list X is a 2-D LoDTensor with shape [N, 4] holds N boxes, "
             "each box is represented as [xmin, ymin, xmax, ymax], "
             "the shape of X is [N, 4]. [xmin, ymin] is the left top "
             "coordinate of the box if the input is image feature map, they "
             "are close to the origin of the coordinate system. "
             "[xmax, ymax] is the right bottom coordinate of the box. "
             "This tensor can contain LoD information to represent a batch "
             "of inputs. One instance of this batch can contain different "
             "numbers of entities.");
    AddInput("Y",
             "(Tensor, default Tensor<float>) "
             "Box list Y holds M boxes, each box is represented as "
             "[xmin, ymin, xmax, ymax], the shape of X is [N, 4]. "
             "[xmin, ymin] is the left top coordinate of the box if the "
             "input is image feature map, and [xmax, ymax] is the right "
             "bottom coordinate of the box.");

    AddOutput("Out",
              "(LoDTensor, the lod is same as input X) The output of "
              "iou_similarity op, a tensor with shape [N, M] "
              "representing pairwise iou scores.");

    AddComment(R"DOC(
**IOU Similarity Operator**

Computes intersection-over-union (IOU) between two box lists.
Box list 'X' should be a LoDTensor and 'Y' is a common Tensor,
boxes in 'Y' are shared by all instance of the batched inputs of X.
Given two boxes A and B, the calculation of IOU is as follows:

$$
IOU(A, B) = 
\\frac{area(A\\cap B)}{area(A)+area(B)-area(A\\cap B)}
$$

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(iou_similarity, ops::IOUSimilarityOp,
                  ops::IOUSimilarityOpMaker,
                  paddle::framework::EmptyGradOpMaker);

REGISTER_OP_CPU_KERNEL(
    iou_similarity,
    ops::IOUSimilarityKernel<paddle::platform::CPUDeviceContext, float>,
    ops::IOUSimilarityKernel<paddle::platform::CPUDeviceContext, double>);
