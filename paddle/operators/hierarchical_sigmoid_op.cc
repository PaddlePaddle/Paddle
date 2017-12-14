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

#include "hierarchical_sigmoid_op.h"

namespace paddle {
namespace operators {

/**
 * Organize the classes into a binary tree. At each node, a sigmoid function
 * is used to calculate the probability of belonging to the right branch.
 * This idea is from "F. Morin, Y. Bengio (AISTATS 05):
 * Hierarchical Probabilistic Neural Network Language Model."
 *
 * Here we uses a simple way of making the binary tree.
 * Assuming the number of classes C = 6,
 * The classes are organized as a binary tree in the following way:
 *
 * @code{.py}
 * *-*-*- 2
 * | | |- 3
 * | |
 * | |-*- 4
 * |   |- 5
 * |
 * |-*- 0
 *   |- 1
 * @endcode
 *
 * where * indicates an internal node, and each leaf node represents a class.
 * - Node 0 ... C-2 are internal nodes.
 * - Node C-1 ... 2C-2 are leaf nodes.
 * - Class c is represented by leaf node \f$c+C-1\f$.
 *
 * We assign an id for each node:
 * - the id of root be 0.
 * - the left child of a node i is 2*i+1.
 * - the right child of a node i is 2*i+2.
 *
 * It's easy to see that:
 * - the parent of node i is \f$\left\lfloor(i-1)/2\right\rfloor\f$.
 * - the j-th level ancestor of node i is
 * \f$\left\lfloor(i+1)/2^{j+1}\right\rfloor - 1\f$.
 * - A node i is a left child of its parent if \f$(i-1)\%2==0\f$.
 *
 */

class HierarchicalSigmoidOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->hasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null.");
    const int64_t batch_size = ctx->GetInputDim("X")[0];
    std::vector<int64_t> output_shape({batch_size, num_classes_ - 1});
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};

class HierarchicalSigmoidGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {}
};

class HierarchicalSigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HierarchicalSigmoidOpMaker(framework::OpProto* proto,
                             framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor, required) The input Tensor, which the shape is"
             "[N * D], which N is the size of mini-batch,"
             "D is the embded size");
    AddInput("Parameters",
             "(Tensor, required), The parameters of hierarchical "
             "sigmoid operator, each of them is s a 3-D tensor, the shape is"
             "[N, num_classes - 1, D]");
    AddInput("Label",
             "(Tensor, required), The labels of training data. It's a"
             "1-D tensor, which the shape is [1, N]");
    AddInput("Bias",
             "(Tensor, optional), The bias is a 1-D tensor, "
             "which is applied to the output, the shape is"
             "[1, num_classes -1]");
    AddOutput("Out",
              "(Tensor, required) The output of hierarchical sigmoid operator."
              "the shape is [N, 1]");
    AddAttr<int>("num_classes", "(int, required)", "The number of classes");
    AddComment(R"DOC(
The hierarchical sigmoid operator organize the classes into a binary tree.
At each node, a sigmoid function is used to caculate the probability of 
belonging to the right branch. This idea is from 
"F. Morin, Y. Bengio (AISTATS 05): 
Hierarchical Probabilistic Neural Network Language Model."
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(hierarchical_sigmoid, ops::HierarchicalSigmoidOp,
            ops::HierarchicalSigmoidOpMaker, hierarchical_sigmoid_grad,
            ops::HierarchicalSigmoidGradOp);
REGISTER_OP_CPU_KERNEL(
    hierarchical_sigmoid,
    ops::HierarchicalSigmoidOpKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    hierarchical_sigmoid_grad,
    ops::HierarchicalSigmoidGradOpKernel<paddle::platform::CPUPlace, float>);
