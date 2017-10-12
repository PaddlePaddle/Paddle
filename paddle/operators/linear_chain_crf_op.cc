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

#include "paddle/operators/linear_chain_crf_op.h"

namespace paddle {
namespace operators {

class LinearChainCrfOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LinearChainCrfOpMaker(framework::OpProto* proto,
                        framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Emission",
        "(LoDTensor, default: LoDTensor<float>). "
        "The unscaled emission weight matrix for the linear chain CRF. "
        "This input is a LoDTensor with shape [N x D] where N is the total "
        "element number of all input squences in a mini-batch, "
        "and D is the total tag number.");
    AddInput(
        "Transition",
        "(Tensor, default: Tensor<float>). A Tensor with shape [(D + 2) x D]. "
        "The learnable parameter for linear_chain_crf operator. "
        "See more details in the operator's comments.");
    AddInput(
        "Label",
        "(LoDTensor, default: LoDTensor<int>). The ground truth which is a 2-D "
        "LoDTensor with shape [N x 1], where N is the total element number in "
        "a mini-batch.");
    AddOutput(
        "Alpha",
        "Tensor, default: Tensor<float>. The forward vectors for the entire "
        "batch. A two dimensional tensor with shape [N x D], "
        "denoted as \f$\alpha\f$. \f$\alpha$\f is a memo table used to "
        "calculate the normalization factor in CRF. \f$\alpha[k, v]$\f stores "
        "the unnormalized probabilites of all possible unfinished sequences of "
        "tags that end at position \f$k$\f with tag \f$v$\f. For each \f$k$\f, "
        "\f$\alpha[k, v]$\f is a vector of length \f$D$\f with a component for "
        "each tag value \f$v$\f. This vector is called a forward vecotr and "
        "will also be used in backward computations.")
        .AsIntermediate();
    AddOutput(
        "LogLikelihood",
        "(Tensor, default: Tensor<float>). The logarithm of the conditional "
        "likelihood of each training sample in a mini-batch. This is a 2-D "
        "tensor with shape [S x 1], where S is the sequence number in a "
        "mini-batch. "
        "Note: S is equal to the sequence number in a mini-batch. The output "
        "is no longer a LoDTensor.");
    AddComment(R"DOC(
Conditional Random Field defines an undirected probabilistic graph with nodes
denoting random variables and edges denoting dependencies between these
variables. CRF learns the conditional probability \f$P(Y|X)\f$, where
\f$X = (x_1, x_2, ... , x_n)\f$ are structured inputs and
\f$Y = (y_1, y_2, ... , y_n)\f$ are labels for the inputs.

Linear chain CRF is a special case of CRF that is useful for sequence labeling
task. Sequence labeling tasks do not assume a lot of conditional
independences among inputs. They only concern about the input and the output
being linear sequences. Thus, the graph model of CRF is a simple chain or
a line, which results in a linear chain CRF.

This operator implements the Forward-Backward algorithm for linear chain CRF.
Please see http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

Equation:

- Denote the first input of this operator (Emission) as \f$x\f$ here.
- The first D values of the second input (Transition) of this operator are for
starting weights, denoted as \f$a\f$ here.
- The next D values of the second input (Transition) of this operator are for
ending weights, denoted as \f$b\f$ here.
- The remaning values of the second input (Transition) are for transition
weights, denoted as \f$w\f$ here.
- Denote the third input of this operator (Label) as \f$s\f$ here.

The probability of a sequence \f$s\f$ of length \f$L\f$ is defined as:
\f$P(s) = (1/Z) exp(a_{s_1} + b_{s_L}
                 + \sum_{l=1}^L x_{s_l}
                 + \sum_{l=2}^L w_{s_{l-1},s_l})\f$
where \f$Z\f$ is a normalization value so that the sum of \f$P(s)\f$ over
all possible sequences is \f$1\f$, and \f$x\f$ is the emission feature weight
to the linear chain CRF.

Finaly, the linear chain CRF operator outputs the logarithm of the conditional
likelihood of each training sample in a mini-batch.

NOTE:
1. The feature function for a CRF is made up of the emission features and the
transition features. The emission feature weights are NOT computed in
this operator. They MUST be computed first before this operator is called.

2. Because this operator performs globally normaliztion over all possible
sequences internally, it expects UNSCALED emission feature weights.
Please do not call this op with the emission feature being output of any
nonlinear activation.

3. The 2nd dimension of the first input of this operator (Emission) MUST be
equal to the tag number.

)DOC");
  }
};

class LinearChainCrfOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
};

class LinearChainCrfGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(linear_chain_crf, ops::LinearChainCrfOp, ops::LinearChainCrfOpMaker,
            linear_chain_crf_grad, ops::LinearChainCrfGradOp);
REGISTER_OP_CPU_KERNEL(linear_chain_crf, ops::LinearChainCrfOpKernel<float>);
REGISTER_OP_CPU_KERNEL(linear_chain_crf_grad,
                       ops::LinearChainCrfGradOpKernel<float>);
