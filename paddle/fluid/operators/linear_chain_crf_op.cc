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

#include "paddle/fluid/operators/linear_chain_crf_op.h"

#include <memory>

namespace paddle {
namespace operators {

class LinearChainCRFOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Emission",
             "(LoDTensor/Tensor<float>). When a LoDTensor input,A 2-D LoDTensor"
             " with shape [N x D], where N is the size of the "
             "mini-batch and D is the total tag number. The unscaled emission "
             "weight matrix for the linear chain CRF. When a Tensor input,"
             "A Tensor with shape [N x S x D], where N is batch number,"
             "S is max length of sequences, D is the total tag number."
             "A LoDTensor or Tensor with type float32, float64.");
    AddInput("Transition",
             "(Tensor, default Tensor<float>) A 2-D Tensor with shape "
             "[(D + 2) x D]. The learnable parameter for the linear_chain_crf "
             "operator. See more details in the operator's comments.");
    AddInput("Label",
             "(LoDTensor/Tensor<int64_t>), when a LoDTensor input,  "
             "[N x 1], where N is the total element number in a mini-batch. "
             "when a Tensor input, [N x S], where N is batch number. "
             "S is max length of sequences. The ground truth."
             "A  LoDTensor or Tensor with int64.");
    AddInput("Length",
             "(Tensor, default Tensor<int64_t>) A Tensor with shape "
             "[M x 1], where M is the sequence number in a mini-batch."
             "A Tensor with type int64.")
        .AsDispensable();
    AddOutput(
        "Alpha",
        "(Tensor, default Tensor<float>), the same shape with Emission. "
        "The forward vectors for the entire batch. Denote it as $\alpha$. "
        "$\alpha$ is a memo table used to calculate the normalization "
        "factor in CRF. $\alpha[k, v]$ stores the unnormalized "
        "probabilites of all possible unfinished sequences of tags that end at "
        "position $k$ with tag $v$. For each $k$, "
        "$\alpha[k, v]$ is a vector of length $D$ with a component for "
        "each tag value $v$. This vector is called a forward vecotr and "
        "will also be used in backward computations.")
        .AsIntermediate();
    AddOutput(
        "EmissionExps",
        "(Tensor, default Tensor<float>), the same shape with Emission. "
        "The exponentials of Input(Emission). This is an intermediate "
        "computational result in forward computation, and will be reused in "
        "backward computation."
        "A LoDTensor or Tensor with type float32, float64.")
        .AsIntermediate();
    AddOutput(
        "TransitionExps",
        "(Tensor, default Tensor<float>) A 2-D Tensor with shape "
        "[(D + 2) x D]. The exponentials of Input(Transition). This is an "
        "intermediate computational result in forward computation, and "
        "will be reused in backward computation."
        "A LoDTensor or Tensor with type float32, float64.")
        .AsIntermediate();
    AddOutput(
        "LogLikelihood",
        "(Tensor, default Tensor<float>) The logarithm of the conditional "
        "likelihood of each training sample in a mini-batch. This is a 2-D "
        "tensor with shape [S x 1], where S is the sequence number in a "
        "mini-batch. Note: S is equal to the sequence number in a mini-batch. "
        "A Tensor with type float32, float64.");
    AddComment(R"DOC(
Conditional Random Field defines an undirected probabilistic graph with nodes
denoting random variables and edges denoting dependencies between these
variables. CRF learns the conditional probability $P(Y|X)$, where
$X = (x_1, x_2, ... , x_n)$ are structured inputs and
$Y = (y_1, y_2, ... , y_n)$ are labels for the inputs.

Linear chain CRF is a special case of CRF that is useful for sequence labeling
task. Sequence labeling tasks do not assume a lot of conditional
independences among inputs. The only constraint they impose is that the input
and output must be linear sequences. Thus, the graph of such a CRF is a simple
chain or a line, which results in the linear chain CRF.

This operator implements the Forward-Backward algorithm for the linear chain
CRF. Please refer to http://www.cs.columbia.edu/~mcollins/fb.pdf and
http://cseweb.ucsd.edu/~elkan/250Bwinter2012/loglinearCRFs.pdf for details.

Equation:

1. Denote Input(Emission) to this operator as $x$ here.
2. The first D values of Input(Transition) to this operator are for starting
weights, denoted as $a$ here.
3. The next D values of Input(Transition) of this operator are for ending
weights, denoted as $b$ here.
4. The remaning values of Input(Transition) are for transition weights,
denoted as $w$ here.
5. Denote Input(Label) as $s$ here.

The probability of a sequence $s$ of length $L$ is defined as:
$$P(s) = (1/Z) \exp(a_{s_1} + b_{s_L}
                + \sum_{l=1}^L x_{s_l}
                + \sum_{l=2}^L w_{s_{l-1},s_l})$$

where $Z$ is a normalization value so that the sum of $P(s)$ over
all possible sequences is 1, and $x$ is the emission feature weight
to the linear chain CRF.

Finally, the linear chain CRF operator outputs the logarithm of the conditional
likelihood of each training sample in a mini-batch.

NOTE:

1. The feature function for a CRF is made up of the emission features and the
transition features. The emission feature weights are NOT computed in
this operator. They MUST be computed first before this operator is called.

2. Because this operator performs global normalization over all possible
sequences internally, it expects UNSCALED emission feature weights.
Please do not call this op with the emission feature being output of any
nonlinear activation.

3. The 2nd dimension of Input(Emission) MUST be equal to the tag number.

)DOC");
  }
};

class LinearChainCRFOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Emission"), "Input", "Emission",
                   "LinearChainCRF");
    OP_INOUT_CHECK(ctx->HasInput("Transition"), "Input", "Transition",
                   "LinearChainCRF");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "LinearChainCRF");

    OP_INOUT_CHECK(ctx->HasOutput("Alpha"), "Output", "Alpha",
                   "LinearChainCRF");
    OP_INOUT_CHECK(ctx->HasOutput("EmissionExps"), "Output", "EmissionExps",
                   "LinearChainCRF");
    OP_INOUT_CHECK(ctx->HasOutput("TransitionExps"), "Output", "TransitionExps",
                   "LinearChainCRF");
    OP_INOUT_CHECK(ctx->HasOutput("LogLikelihood"), "Output", "LogLikelihood",
                   "LinearChainCRF");

    auto transition_dims = ctx->GetInputDim("Transition");
    PADDLE_ENFORCE_EQ(transition_dims.size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The Input(Transition) should be a 2-D tensor. But "
                          "received: input rank %u, input shape [%s].",
                          transition_dims.size(), transition_dims));
    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (transition_dims[0] <= 0 || transition_dims[1] <= 0)) {
      check = false;
    }
    if (check) {
      PADDLE_ENFORCE_EQ(
          transition_dims[0] - 2, transition_dims[1],
          platform::errors::InvalidArgument(
              "An invalid dimension for the Input(Transition), which should "
              "be a 2-D tensor with shape [(D + 2) x D]. But received: input "
              "rank %u, "
              "input shape [%s].",
              transition_dims.size(), transition_dims));
    }
    auto emission_dims = ctx->GetInputDim("Emission");
    if (ctx->HasInput("Length")) {
      PADDLE_ENFORCE_EQ(emission_dims.size(), 3,
                        platform::errors::InvalidArgument(
                            "The Input(Emission) should be a 3-D tensor. But "
                            "received: input rank %u, input shape [%s].",
                            emission_dims.size(), emission_dims));
      auto label_dims = ctx->GetInputDim("Label");
      PADDLE_ENFORCE_EQ(
          (label_dims.size() == 3UL && label_dims[2] == 1) ||
              (label_dims.size() == 2UL),
          true,
          platform::errors::InvalidArgument(
              "The Input(Label) should be a 3-D tensor with last dimension "
              "fixed to 1 or a 2-D tensor in padding mode. But received: input "
              "rank %u, input shape [%s].",
              label_dims.size(), label_dims));
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(emission_dims[0], label_dims[0],
                          platform::errors::InvalidArgument(
                              "The batch size of Input(Emission) "
                              "and Input(Label) should be the same. But "
                              "received Input(Emission): "
                              "rank %u, shape [%s]; received Input(Label): "
                              "rank %u, shape [%s].",
                              emission_dims.size(), emission_dims,
                              label_dims.size(), label_dims));
        PADDLE_ENFORCE_EQ(emission_dims[1], label_dims[1],
                          platform::errors::InvalidArgument(
                              "The max length of Input(Emission) "
                              "and Input(Label) should be the same. But "
                              "received Input(Emission): "
                              "rank %u, shape [%s]; received Input(Label): "
                              "rank %u, shape [%s].",
                              emission_dims.size(), emission_dims,
                              label_dims.size(), label_dims));
      }
    } else {
      PADDLE_ENFORCE_EQ(
          emission_dims.size(), 2,
          platform::errors::InvalidArgument(
              "The Input(Emission) should be a 2-D tensor. But received: "
              "input rank %u, input shape [%s].",
              emission_dims.size(), emission_dims));
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(emission_dims[1], transition_dims[1],
                          platform::errors::InvalidArgument(
                              "The 2nd dimension of the Input(Emission) and "
                              "the Input(Transition) "
                              "should be equal to the tag number. But received "
                              "Input(Emission): rank "
                              "%u, shape [%s]; received Input(Transition): "
                              "rank %u, shape [%s].",
                              emission_dims.size(), emission_dims,
                              transition_dims.size(), transition_dims));
      }

      auto label_dims = ctx->GetInputDim("Label");
      PADDLE_ENFORCE_EQ(
          label_dims.size(), 2,
          platform::errors::InvalidArgument(
              "The Input(Label) should be a 2-D tensor with the 2nd "
              "dimensions fixed to 1. But received: input rank %u, "
              "input shape [%s].",
              label_dims.size(), label_dims));
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            emission_dims[0], label_dims[0],
            platform::errors::InvalidArgument(
                "The first dimension of Input(Emission) and Input(Label) "
                "should be the same. But received Input(Emission): rank %u, "
                "shape "
                "[%s]; received Input(Label): rank %u, shape [%s].",
                emission_dims.size(), emission_dims, label_dims.size(),
                label_dims));
      }
    }
    ctx->SetOutputDim("Alpha", emission_dims);
    ctx->SetOutputDim("EmissionExps", emission_dims);
    ctx->SetOutputDim("TransitionExps", transition_dims);
    // TODO(caoying) This is tricky. The 1st dimension of Output(LogLikelihood)
    // is the sequence number in a mini-batch. The dimension set here should be
    // resized to its correct size in the function Compute. Fix this once we can
    // get LoD information in the InferShape interface.
    ctx->SetOutputDim("LogLikelihood", {emission_dims[0], 1});
  }

 protected:
  // Explicitly set that the data type of computation kernel of linear_chain_crf
  // is determined by its input "Emission".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Emission"),
        platform::CPUPlace());
  }
};

class LinearChainCRFGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("EmissionExps"), "Input", "EmissionExps",
                   "LinearChainCRFGrad");
    OP_INOUT_CHECK(ctx->HasInput("TransitionExps"), "Input", "TransitionExps",
                   "LinearChainCRFGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("LogLikelihood")),
                   "Input", framework::GradVarName("LogLikelihood"),
                   "LinearChainCRFGrad");

    auto transition_exps_dims = ctx->GetInputDim("TransitionExps");
    auto emission_exps_dims = ctx->GetInputDim("EmissionExps");
    if (ctx->HasOutput(framework::GradVarName("Emission"))) {
      ctx->SetOutputDim(framework::GradVarName("Emission"), emission_exps_dims);
      if (ctx->HasInput("Length") == false) {
        ctx->ShareLoD("Emission", framework::GradVarName("Emission"));
      }
    }

    if (ctx->HasOutput(framework::GradVarName("Transition"))) {
      ctx->SetOutputDim(framework::GradVarName("Transition"),
                        transition_exps_dims);
      ctx->ShareLoD("Transition", framework::GradVarName("Transition"));
    }
  }

 protected:
  // Explicitly set that the data type of output of the linear_chain_crf_grad
  // operator is determined by its input: gradients of LogLikelihood.
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(
            ctx, framework::GradVarName("LogLikelihood")),
        platform::CPUPlace());
  }
};

template <typename T>
class LinearChainCRFGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("linear_chain_crf_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("Emission", this->Input("Emission"));
    op->SetInput("Transition", this->Input("Transition"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("Alpha", this->Output("Alpha"));
    op->SetInput("EmissionExps", this->Output("EmissionExps"));
    op->SetInput("TransitionExps", this->Output("TransitionExps"));
    if (this->HasInput("Length")) {
      op->SetInput("Length", this->Input("Length"));
    }
    op->SetInput(framework::GradVarName("LogLikelihood"),
                 this->OutputGrad("LogLikelihood"));

    op->SetOutput(framework::GradVarName("Emission"),
                  this->InputGrad("Emission"));
    op->SetOutput(framework::GradVarName("Transition"),
                  this->InputGrad("Transition"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(LinearChainCRFGradNoNeedBufferVarsInference,
                                    "Transition", "Emission");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(linear_chain_crf, ops::LinearChainCRFOp,
                  ops::LinearChainCRFOpMaker,
                  ops::LinearChainCRFGradMaker<paddle::framework::OpDesc>,
                  ops::LinearChainCRFGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(linear_chain_crf_grad, ops::LinearChainCRFGradOp,
                  ops::LinearChainCRFGradNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf,
    ops::LinearChainCRFOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LinearChainCRFOpKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    linear_chain_crf_grad,
    ops::LinearChainCRFGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LinearChainCRFGradOpKernel<paddle::platform::CPUDeviceContext,
                                    double>);
